from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class EvalResults:
	loss: float
	ppl: float
	bpc: float
	token_acc: float
	topk_acc: Dict[int, float]
	n_tokens: int
	n_batches: int


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
	return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _valid_mask(targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
	return (targets != ignore_index)


def _compute_batch_metrics(
	logits: torch.Tensor,
	targets: torch.Tensor,
	ignore_index: int,
	topk: Sequence[int] = (1, 5),
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
	B, T, V = logits.shape
	logits_flat = logits.view(-1, V)
	targets_flat = targets.view(-1)

	valid = _valid_mask(targets_flat, ignore_index)
	n_valid = valid.sum()
	loss_sum = torch.tensor(0.0, device=logits.device)
	correct_1 = torch.tensor(0, device=logits.device)
	correct_k: Dict[int, torch.Tensor] = {k: torch.tensor(0, device=logits.device) for k in topk}

	if n_valid > 0:
		loss = F.cross_entropy(logits_flat[valid], targets_flat[valid], reduction="mean")
		loss_sum = loss * n_valid.float()

		pred_1 = logits_flat.argmax(dim=-1)
		correct_1 = (pred_1.eq(targets_flat) & valid).sum()

		max_k = max(topk) if len(topk) > 0 else 0
		if max_k > 1:
			topk_idx = torch.topk(logits_flat, k=max_k, dim=-1).indices  # [N, max_k]
			for k in topk:
				hits_k = (topk_idx[:, :k] == targets_flat.unsqueeze(-1)).any(dim=-1)
				correct_k[k] = (hits_k & valid).sum()
		else:
			if 1 in correct_k:
				correct_k[1] = correct_1

	return loss_sum, correct_1, correct_k


@torch.no_grad()
def evaluate_model(
	model: torch.nn.Module,
	dataloader: Iterable[Mapping[str, torch.Tensor]],
	device: Optional[torch.device] = None,
	*,
	ignore_index: int = -100,
	topk: Sequence[int] = (1, 5),
	autocast_dtype: Optional[torch.dtype] = None,
	max_batches: Optional[int] = None,
) -> EvalResults:
	model_was_training = model.training
	model.eval()

	if device is None:
		try:
			device = next(model.parameters()).device
		except StopIteration:
			device = torch.device("cpu")

	total_loss_sum = torch.tensor(0.0, device=device)
	total_tokens = torch.tensor(0, device=device)
	total_correct_1 = torch.tensor(0, device=device)
	total_correct_k = {k: torch.tensor(0, device=device) for k in topk}
	n_batches = 0

	autocast_cm = (
		torch.amp.autocast(dtype=autocast_dtype, device_type=device.type)
		if autocast_dtype is not None else None
	)

	def _loop_cm():
		return autocast_cm if autocast_cm is not None else torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_dtype is not None else torch.enable_grad()  # dummy, will be used in a 'with' context

	for i, batch in enumerate(dataloader):
		if max_batches is not None and i >= max_batches:
			break
		n_batches += 1
		b = _to_device(batch, device)

		fwd_kwargs: Dict[str, torch.Tensor] = {}
		if getattr(model, "use_external_embeddings", False):
			assert "x_external" in b, "Batch missing 'x_external' for adapter path"
			fwd_kwargs["x_external"] = b["x_external"]
		else:
			assert "x_tokens" in b, "Batch missing 'x_tokens' for token path"
			fwd_kwargs["x_tokens"] = b["x_tokens"]

		if "key_padding_mask" in b:
			fwd_kwargs["key_padding_mask"] = b["key_padding_mask"]
		if "self_attn_mask" in b:
			fwd_kwargs["self_attn_mask"] = b["self_attn_mask"]

		targets = b["targets"]

		if autocast_cm is not None:
			with autocast_cm:
				logits, _ = model(**fwd_kwargs)
		else:
			logits, _ = model(**fwd_kwargs)

		loss_sum, correct_1, correct_k = _compute_batch_metrics(logits, targets, ignore_index, topk)
		valid = _valid_mask(targets.view(-1), ignore_index).sum()

		total_loss_sum += loss_sum
		total_tokens += valid
		total_correct_1 += correct_1
		for k in topk:
			total_correct_k[k] += correct_k[k]

	n_tok = int(total_tokens.item()) if total_tokens.numel() > 0 else 0
	mean_loss = float((total_loss_sum / total_tokens).item()) if n_tok > 0 else float("nan")
	ppl = math.exp(mean_loss) if n_tok > 0 else float("nan")
	bpc = mean_loss / math.log(2) if n_tok > 0 else float("nan")

	acc1 = float((total_correct_1 / total_tokens).item()) if n_tok > 0 else float("nan")
	topk_acc = {k: float((total_correct_k[k] / total_tokens).item()) if n_tok > 0 else float("nan") for k in topk}

	if model_was_training:
		model.train()

	return EvalResults(
		loss=mean_loss,
		ppl=ppl,
		bpc=bpc,
		token_acc=acc1,
		topk_acc=topk_acc,
		n_tokens=n_tok,
		n_batches=n_batches,
	)


@torch.no_grad()
def evaluate_checkpoint(
	model: torch.nn.Module,
	state_dict: Mapping[str, torch.Tensor],
	dataloader: Iterable[Mapping[str, torch.Tensor]],
	device: Optional[torch.device] = None,
	*,
	strict: bool = True,
	**eval_kwargs,
) -> EvalResults:
	missing, unexpected = model.load_state_dict(state_dict, strict=strict)
	if not strict:
		pass
	if device is None:
		try:
			device = next(model.parameters()).device
		except StopIteration:
			device = torch.device("cpu")
	model.to(device)
	return evaluate_model(model, dataloader, device=device, **eval_kwargs)

