from __future__ import annotations

import argparse
import os
import time
from typing import Optional
import csv
from datetime import datetime, UTC

"""Configure cache directories early (before importing torch) to avoid /tmp space issues."""
try:
	_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	_default_cache = os.path.join(_root_dir, "checkpoints", ".inductor_cache")
	os.makedirs(_default_cache, exist_ok=True)
	_triton_dir = os.path.join(_default_cache, "triton"); os.makedirs(_triton_dir, exist_ok=True)
	_tmp_dir = os.path.join(_default_cache, "tmp"); os.makedirs(_tmp_dir, exist_ok=True)
	_nvfuser_dir = os.path.join(_default_cache, "nvfuser"); os.makedirs(_nvfuser_dir, exist_ok=True)
	_cuda_rtc_dir = os.path.join(_default_cache, "cuda-rtc"); os.makedirs(_cuda_rtc_dir, exist_ok=True)
	os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _default_cache)
	os.environ.setdefault("TRITON_CACHE_DIR", _triton_dir)
	os.environ.setdefault("PYTORCH_NVFUSER_CACHE_DIR", _nvfuser_dir)
	os.environ.setdefault("CUDA_CACHE_PATH", _cuda_rtc_dir)
	os.environ.setdefault("TMPDIR", _tmp_dir)
	os.environ.setdefault("XDG_CACHE_HOME", _default_cache)
except Exception:
	pass

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.data_loader import DataLoader as MemmapLoader
from model import LMModel
from eval import evaluate_model
from embeddings.external_embedder import GGUFEmbedder


def human_time(seconds: float) -> str:
	seconds = int(seconds)
	h = seconds // 3600
	m = (seconds % 3600) // 60
	s = seconds % 60
	if h > 0:
		return f"{h}h {m:02d}m {s:02d}s"
	if m > 0:
		return f"{m}m {s:02d}s"
	return f"{s}s"


def maybe_compile(model: nn.Module, do_compile: bool, backend: str = "inductor") -> nn.Module:
	if do_compile and hasattr(torch, "compile"):
		try:
			return torch.compile(model, mode="max-autotune", backend=backend)
		except Exception as e:
			print(f"Warning: torch.compile failed with backend={backend}: {e}. Falling back to eager.")
			return model
	return model


def get_optimizer(params, lr: float, weight_decay: float, betas=(0.9, 0.95)):
	fused = torch.cuda.is_available()
	try:
		return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, fused=fused)
	except TypeError:
		return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def set_fastmat():
	torch.backends.cudnn.benchmark = True
	if torch.cuda.is_available():
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
	try:
		torch.set_float32_matmul_precision("high")
	except Exception:
		pass


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, best_val: Optional[float]):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save({
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"step": step,
		"best_val": best_val,
	}, path)


class CSVLogger:
	"""Simple CSV logger that writes one row per call with a fixed schema."""

	def __init__(self, path: str, fieldnames: list[str]):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		file_exists = os.path.exists(path)
		self._f = open(path, mode="a", newline="")
		self._writer = csv.DictWriter(self._f, fieldnames=fieldnames)
		if not file_exists:
			self._writer.writeheader()
			self._f.flush()

	def log(self, row: dict):
		# Ensure all fields are strings or numbers, and add timestamp if missing
		if "timestamp" not in row:
			row["timestamp"] = datetime.now(UTC).isoformat()
		self._writer.writerow(row)
		self._f.flush()

	def close(self):
		try:
			self._f.close()
		except Exception:
			pass


def main():
	parser = argparse.ArgumentParser(description="Train LMModel on memmap data with AMP and ETA.")
	# Data
	parser.add_argument("--train_file", type=str, default="data/tokenized/train.bin")
	parser.add_argument("--val_file", type=str, default="data/tokenized/val.bin")
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--block_size", type=int, default=1024)
	# Model
	parser.add_argument("--vocab_size", type=int, default=50304)
	parser.add_argument("--n_layer", type=int, default=12)
	parser.add_argument("--n_head", type=int, default=12)
	parser.add_argument("--n_embd", type=int, default=768)
	parser.add_argument("--dropout", type=float, default=0.0)
	parser.add_argument("--kv_lora_rank", type=int, default=256)
	parser.add_argument("--down_rate", type=int, default=2)
	# External embeddings
	parser.add_argument("--use_external_embeddings", action="store_true")
	parser.add_argument("--embed_model_path", type=str, default="models/Mungert/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-iq3_m.gguf")
	parser.add_argument("--ext_emb_dim", type=int, default=0, help="If 0, inferred from embedding model")
	parser.add_argument("--adapter_bottleneck", type=int, default=64)
	# Optimization
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--weight_decay", type=float, default=0.1)
	parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
	parser.add_argument("--grad_accum_steps", type=int, default=1)
	parser.add_argument("--max_steps", type=int, default=2000)
	parser.add_argument("--epochs", type=int, default=0, help="If > 0, set max_steps = epochs * steps_per_epoch computed from dataset size")
	parser.add_argument("--clip_grad", type=float, default=1.0)
	parser.add_argument("--use_8bit_optim", action="store_true", help="Use bitsandbytes Adam8bit if available")
	# Runtime
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--compile", action="store_true")
	parser.add_argument("--compile_backend", type=str, default="inductor", choices=["inductor", "aot_eager"], help="compile backend")
	parser.add_argument("--seed", type=int, default=1337)
	parser.add_argument("--out_dir", type=str, default="checkpoints")
	parser.add_argument("--log_csv", type=str, default=None, help="Path to CSV log file; defaults to <out_dir>/training_log.csv")
	parser.add_argument("--eval_every", type=int, default=200)
	parser.add_argument("--save_every", type=int, default=500)
	parser.add_argument("--log_every", type=int, default=10)
	parser.add_argument("--log_csv_every", type=int, default=10, help="Write train_step rows to CSV every N steps")
	parser.add_argument("--val_batches", type=int, default=50, help="Batches to use for validation evaluation")
	parser.add_argument("--gradient_checkpoint", action="store_true")
	parser.add_argument("--prefetch", action="store_true", help="Overlap H2D copies with compute via CUDA prefetcher")
	parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel training (use with torchrun)")
	parser.add_argument("--dist_backend", type=str, default="nccl")
	parser.add_argument("--local_rank", type=int, default=-1)
	parser.add_argument("--inductor_cache_dir", type=str, default=None, help="Directory for TorchInductor/Triton cache (defaults to <out_dir>/.inductor_cache)")

	args = parser.parse_args()

	torch.manual_seed(args.seed)
	# Reduce allocator fragmentation if not set by user (capital True for compatibility)
	os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
	set_fastmat()

	# Optional DDP init (use torchrun)
	world_size = int(os.environ.get("WORLD_SIZE", "1"))
	is_distributed = args.ddp and world_size > 1 and torch.cuda.is_available()
	local_rank_env = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
	if is_distributed:
		assert args.device.startswith("cuda"), "DDP expects CUDA devices"
		torch.cuda.set_device(local_rank_env)
		device = torch.device(f"cuda:{local_rank_env}")
		dist.init_process_group(backend=args.dist_backend, init_method="env://")
		rank = dist.get_rank()
		is_main = rank == 0
	else:
		device = torch.device(args.device)
		rank = 0
		is_main = True

	# Data
	# Use CPU + pinned memory for prefetching when fast path (no external embeddings)
	if args.prefetch and device.type == "cuda" and not args.use_external_embeddings:
		train_loader = MemmapLoader(args.train_file, batch_size=args.batch_size, block_size=args.block_size, device="cpu", pin_memory=True)
		val_loader = None
		if os.path.exists(args.val_file) and os.path.getsize(args.val_file) > 0:
			val_loader = MemmapLoader(args.val_file, batch_size=args.batch_size, block_size=args.block_size, device="cpu", pin_memory=True)
	else:
		train_loader = MemmapLoader(args.train_file, batch_size=args.batch_size, block_size=args.block_size, device=args.device)
		val_loader = None
		if os.path.exists(args.val_file) and os.path.getsize(args.val_file) > 0:
			val_loader = MemmapLoader(args.val_file, batch_size=args.batch_size, block_size=args.block_size, device=args.device)

	# Optional external embedder; initialize before model to infer embedding dim
	embedder = None
	if args.use_external_embeddings:
		embedder = GGUFEmbedder(args.embed_model_path)
		if args.ext_emb_dim == 0:
			args.ext_emb_dim = embedder.dim

	# Model
	model = LMModel(
		vocab_size=args.vocab_size,
		block_size=args.block_size,
		n_layer=args.n_layer,
		n_head=args.n_head,
		n_embd=args.n_embd,
		use_external_embeddings=args.use_external_embeddings,
		ext_emb_dim=args.ext_emb_dim if args.use_external_embeddings else 768,
		adapter_bottleneck=args.adapter_bottleneck,
		dropout=args.dropout,
		kv_lora_rank=args.kv_lora_rank,
		down_rate=args.down_rate,
		gradient_checkpoint=args.gradient_checkpoint,
	).to(device)

	# Configure TorchInductor/Triton cache to a writable location with space
	if args.compile:
		cache_dir = args.inductor_cache_dir or os.environ.get("TORCHINDUCTOR_CACHE_DIR") or os.path.join(args.out_dir, ".inductor_cache")
		try:
			os.makedirs(cache_dir, exist_ok=True)
			triton_dir = os.path.join(cache_dir, "triton")
			tmp_dir = os.path.join(cache_dir, "tmp")
			nvfuser_dir = os.path.join(cache_dir, "nvfuser")
			cuda_rtc_dir = os.path.join(cache_dir, "cuda-rtc")
			for d in (triton_dir, tmp_dir, nvfuser_dir, cuda_rtc_dir):
				os.makedirs(d, exist_ok=True)
			# TorchInductor
			os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
			# Triton cache
			os.environ["TRITON_CACHE_DIR"] = triton_dir
			# NVFuser cache (used by some graph fusers)
			os.environ["PYTORCH_NVFUSER_CACHE_DIR"] = nvfuser_dir
			# CUDA NVRTC cache location
			os.environ["CUDA_CACHE_PATH"] = cuda_rtc_dir
			# Generic temp dirs used by Python and subprocesses
			os.environ["TMPDIR"] = tmp_dir
			os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
			# Try to ensure -lcuda resolves: if only libcuda.so.1 exists, make a local symlink
			candidate_dirs = [
				"/usr/lib/x86_64-linux-gnu",
				"/usr/lib/wsl/lib",
				"/usr/lib64",
				"/usr/lib",
			]
			cuda_so1 = None
			for d in candidate_dirs:
				p = os.path.join(d, "libcuda.so.1")
				if os.path.exists(p):
					cuda_so1 = p
					break
			if cuda_so1 is not None:
				local_lib = os.path.join(cache_dir, "lib")
				os.makedirs(local_lib, exist_ok=True)
				link_target = os.path.join(local_lib, "libcuda.so")
				try:
					if not os.path.exists(link_target):
						os.symlink(cuda_so1, link_target)
					# Help the Triton build find it
					os.environ["LD_LIBRARY_PATH"] = f"{local_lib}:{os.environ.get('LD_LIBRARY_PATH','')}"
					os.environ["LIBRARY_PATH"] = f"{local_lib}:{os.environ.get('LIBRARY_PATH','')}"
				except Exception:
					pass
		except Exception as e:
			print(f"Warning: failed to set inductor cache dir to {cache_dir}: {e}")

	model = maybe_compile(model, args.compile, backend=args.compile_backend)

	# Wrap with DDP if requested
	if is_distributed:
		model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)

	# Optimizer & AMP
	# Optimizer: try 8-bit to cut memory if requested
	if args.use_8bit_optim:
		try:
			import bitsandbytes as bnb  # type: ignore

			optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay)
		except Exception:
			optimizer = get_optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas))
	else:
		optimizer = get_optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas))

	# Prefer BF16 if supported (saves memory vs FP32, more stable than FP16)
	amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
	scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and amp_dtype == torch.float16))

	# Optionally compute steps from epochs
	if args.epochs and args.epochs > 0:
		tokens_per_step_cfg = args.batch_size * args.block_size * args.grad_accum_steps
		try:
			num_tokens = os.path.getsize(args.train_file) // 2  # uint16
		except Exception:
			num_tokens = 0
		steps_per_epoch = max(1, num_tokens // max(1, tokens_per_step_cfg))
		args.max_steps = steps_per_epoch * args.epochs
		print(f"Computed steps: epochs={args.epochs}, steps_per_epoch={steps_per_epoch}, max_steps={args.max_steps}")
	else:
		steps_per_epoch = 0

	# Training loop
	best_val = None
	model.train()

	# Throughput tracking
	tokens_per_step = args.batch_size * args.block_size * args.grad_accum_steps
	cumulative_tokens = 0
	toks_per_s_ema = None
	class _DummyPbar:
		def update(self, *a, **k):
			pass
		def set_postfix(self, *a, **k):
			pass
		def write(self, *a, **k):
			if is_main:
				print(*a)
		def close(self):
			pass

	pbar = tqdm(total=args.max_steps, desc="train", dynamic_ncols=True) if is_main else _DummyPbar()
	start_time = time.time()
	step_time_ema = None

	# CSV Logger (rank0 only)
	csv_path = args.log_csv or os.path.join(args.out_dir, "training_log.csv")
	csv_fields = [
		"timestamp",
		"run_id",
		"step",
		"split",
		"loss",
		"ppl",
		"bpc",
		"acc@1",
		"acc@5",
		"toks_per_s",
		"lr",
		"grad_norm",
		"step_time",
		"cumulative_tokens",
		"gpu_mem_allocated_mb",
		"gpu_mem_reserved_mb",
		"event",
		"batch_size",
		"block_size",
		"n_layer",
		"n_head",
		"n_embd",
		"down_rate",
		"use_external_embeddings",
		"epochs",
		"steps_per_epoch",
	]
	csv_logger = CSVLogger(csv_path, csv_fields) if is_main else None
	run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
	# Log run start metadata
	try:
		gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
	except Exception:
		gpu_name = "unknown"
	if is_main:
		csv_logger.log({
		"run_id": run_id,
		"split": "meta",
		"event": "run_start",
		"batch_size": args.batch_size,
		"block_size": args.block_size,
		"n_layer": args.n_layer,
		"n_head": args.n_head,
		"n_embd": args.n_embd,
		"down_rate": args.down_rate,
		"use_external_embeddings": bool(args.use_external_embeddings),
		"epochs": int(args.epochs),
		"steps_per_epoch": int(steps_per_epoch),
		"lr": float(args.lr),
		"gpu_mem_allocated_mb": float(torch.cuda.memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0,
		"gpu_mem_reserved_mb": float(torch.cuda.memory_reserved() / (1024 * 1024)) if device.type == "cuda" else 0.0,
		"acc@1": None,
		"acc@5": None,
		"ppl": None,
		"bpc": None,
		"toks_per_s": None,
		"grad_norm": None,
		"cumulative_tokens": 0,
		"step": 0,
		"loss": None,
		})

	# Prefetcher (CUDA only, token path)
	class CUDAPrefetcher:
		def __init__(self, loader: MemmapLoader, device: torch.device):
			self.loader = loader
			self.device = device
			self.stream = torch.cuda.Stream(device=device)
			self.next_x = None
			self.next_y = None
			self._preload()

		def _preload(self):
			x_cpu, y_cpu = self.loader.get_batch()
			with torch.cuda.stream(self.stream):
				self.next_x = x_cpu.to(self.device, dtype=torch.long, non_blocking=True)
				self.next_y = y_cpu.to(self.device, dtype=torch.long, non_blocking=True)

		def next(self):
			torch.cuda.current_stream(self.device).wait_stream(self.stream)
			x = self.next_x
			y = self.next_y
			self._preload()
			return x, y

	prefetcher = None
	if args.prefetch and device.type == "cuda" and not args.use_external_embeddings:
		prefetcher = CUDAPrefetcher(train_loader, device)

	for step in range(args.max_steps):
		optimizer.zero_grad(set_to_none=True)
		micro_start = time.time()
		successful_micros = 0

		# Gradient accumulation
		for micro in range(args.grad_accum_steps):
			if prefetcher is not None:
				x, y = prefetcher.next()
			else:
				x, y = train_loader.get_batch()
			try:
				with torch.amp.autocast(enabled=(device.type == "cuda"), device_type=device.type, dtype=amp_dtype):
					if args.use_external_embeddings:
						# WARNING: external embeddings are very slow; prefer disabling for training
						texts = [" ".join(map(str, seq.tolist())) for seq in x.detach().cpu()]
						x_emb = embedder.embed(texts).to(device)  # [B, D]
						x_emb = x_emb.unsqueeze(1).repeat(1, x.size(1), 1)  # [B, T, D]
						mdl = model.module if isinstance(model, DDP) else model
						logits, loss = mdl(x_external=x_emb, targets=y)
					else:
						logits, loss = model(x_tokens=x, targets=y)
					loss_to_backward = loss / args.grad_accum_steps
				scaler.scale(loss_to_backward).backward()
				successful_micros += 1
			except RuntimeError as e:
				if "out of memory" in str(e).lower():
					torch.cuda.empty_cache()
					tqdm.write("OOM encountered; skipping this micro-batch")
					# Log the OOM event
					try:
						allocated = torch.cuda.memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
						reserved = torch.cuda.memory_reserved() / (1024 * 1024) if device.type == "cuda" else 0.0
					except Exception:
						allocated = reserved = 0.0
					if is_main and csv_logger is not None:
						csv_logger.log({
							"step": step + 1,
							"split": "train",
							"loss": float(loss.item()) if 'loss' in locals() else None,
							"event": "oom_skip",
							"gpu_mem_allocated_mb": allocated,
							"gpu_mem_reserved_mb": reserved,
						})
					continue
				else:
					raise

		# Step
		grad_norm_val = None
		if args.clip_grad is not None and args.clip_grad > 0:
			scaler.unscale_(optimizer)
			grad_norm_val = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad))
		scaler.step(optimizer)
		scaler.update()

		# Speed/ETA
		step_time = time.time() - micro_start
		step_time_ema = step_time if step_time_ema is None else 0.9 * step_time_ema + 0.1 * step_time
		steps_left = args.max_steps - (step + 1)
		eta_sec = step_time_ema * steps_left
		tokens_this_step = args.batch_size * args.block_size * successful_micros
		cumulative_tokens += tokens_this_step
		toks_per_s_inst = tokens_this_step / max(step_time, 1e-8)
		toks_per_s_ema = toks_per_s_inst if toks_per_s_ema is None else 0.9 * toks_per_s_ema + 0.1 * toks_per_s_inst

		if is_main and (step + 1) % args.log_every == 0:
			pbar.set_postfix({
				"loss": f"{loss.item():.3f}",
				"toks/s": f"{toks_per_s_ema:,.0f}",
				"ETA": human_time(eta_sec),
			})

		# CSV log for training step
		try:
			allocated = torch.cuda.memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
			reserved = torch.cuda.memory_reserved() / (1024 * 1024) if device.type == "cuda" else 0.0
		except Exception:
			allocated = reserved = 0.0
		if is_main and ((step + 1) % max(1, args.log_csv_every) == 0):
			csv_logger.log({
			"step": step + 1,
			"run_id": run_id,
			"split": "train",
			"loss": float(loss.item()) if 'loss' in locals() else None,
			"ppl": None,
			"bpc": None,
			"acc@1": None,
			"acc@5": None,
			"toks_per_s": float(toks_per_s_ema) if toks_per_s_ema is not None else None,
			"lr": float(args.lr),
			"grad_norm": grad_norm_val,
			"step_time": float(step_time),
			"cumulative_tokens": int(cumulative_tokens),
			"gpu_mem_allocated_mb": float(allocated),
			"gpu_mem_reserved_mb": float(reserved),
			"event": "train_step",
			"batch_size": args.batch_size,
			"block_size": args.block_size,
			"n_layer": args.n_layer,
			"n_head": args.n_head,
			"n_embd": args.n_embd,
			"down_rate": args.down_rate,
			"use_external_embeddings": bool(args.use_external_embeddings),
			"epochs": int(args.epochs),
			"steps_per_epoch": int(steps_per_epoch),
			})

		pbar.update(1)

		# Eval
		if is_main and val_loader is not None and (step + 1) % args.eval_every == 0:
			model.eval()

			# Build a simple iterable for evaluator
			def val_iter(n_batches: int):
				for _ in range(n_batches):
					vx, vy = val_loader.get_batch()
					if args.use_external_embeddings:
						texts = [" ".join(map(str, seq.tolist())) for seq in vx.cpu()]
						x_emb = embedder.embed(texts).to(device).unsqueeze(1).repeat(1, vx.size(1), 1)
						yield {"x_external": x_emb, "targets": vy}
					else:
						yield {"x_tokens": vx, "targets": vy}

			mdl_for_eval = model.module if isinstance(model, DDP) else model
			results = evaluate_model(mdl_for_eval, val_iter(args.val_batches), device=device)
			pbar.write(
				f"val: loss={results.loss:.3f} ppl={results.ppl:.2f} bpc={results.bpc:.3f} acc@1={results.token_acc:.3f} acc@5={results.topk_acc.get(5, float('nan')):.3f}"
			)
			# CSV log for validation
			csv_logger.log({
				"step": step + 1,
				"run_id": run_id,
				"split": "val",
				"loss": float(results.loss),
				"ppl": float(results.ppl),
				"bpc": float(results.bpc),
				"acc@1": float(results.token_acc),
				"acc@5": float(results.topk_acc.get(5, float('nan'))),
				"toks_per_s": None,
				"lr": float(args.lr),
				"grad_norm": None,
				"step_time": None,
				"cumulative_tokens": int(cumulative_tokens),
				"gpu_mem_allocated_mb": float(allocated),
				"gpu_mem_reserved_mb": float(reserved),
				"event": "eval",
				"batch_size": args.batch_size,
				"block_size": args.block_size,
				"n_layer": args.n_layer,
				"n_head": args.n_head,
				"n_embd": args.n_embd,
				"down_rate": args.down_rate,
				"use_external_embeddings": bool(args.use_external_embeddings),
				"epochs": int(args.epochs),
				"steps_per_epoch": int(steps_per_epoch),
			})
			if best_val is None or results.loss < best_val:
				best_val = results.loss
				# Save underlying module when in DDP
				save_checkpoint(os.path.join(args.out_dir, "best.pt"), model.module if isinstance(model, DDP) else model, optimizer, step + 1, best_val)
				pbar.write("saved best checkpoint")
				csv_logger.log({
					"step": step + 1,
					"run_id": run_id,
					"split": "val",
					"loss": float(results.loss),
					"event": "checkpoint_best_saved",
					"cumulative_tokens": int(cumulative_tokens),
					"batch_size": args.batch_size,
					"block_size": args.block_size,
					"n_layer": args.n_layer,
					"n_head": args.n_head,
					"n_embd": args.n_embd,
					"down_rate": args.down_rate,
					"use_external_embeddings": bool(args.use_external_embeddings),
					"epochs": int(args.epochs),
					"steps_per_epoch": int(steps_per_epoch),
				})

			model.train()

		# Save
		if is_main and (step + 1) % args.save_every == 0:
			save_checkpoint(os.path.join(args.out_dir, f"step_{step+1}.pt"), model.module if isinstance(model, DDP) else model, optimizer, step + 1, best_val)
			pbar.write("checkpoint saved")
			csv_logger.log({
				"step": step + 1,
				"run_id": run_id,
				"split": "train",
				"event": "checkpoint_saved",
				"cumulative_tokens": int(cumulative_tokens),
				"batch_size": args.batch_size,
				"block_size": args.block_size,
				"n_layer": args.n_layer,
				"n_head": args.n_head,
				"n_embd": args.n_embd,
				"down_rate": args.down_rate,
				"use_external_embeddings": bool(args.use_external_embeddings),
				"epochs": int(args.epochs),
				"steps_per_epoch": int(steps_per_epoch),
			})

	pbar.close()
	total_time = time.time() - start_time
	if is_main:
		print(f"Training completed in {human_time(total_time)}")
		csv_logger.log({
		"step": args.max_steps,
		"run_id": run_id,
		"split": "train",
		"event": "training_completed",
		"cumulative_tokens": int(cumulative_tokens),
		"batch_size": args.batch_size,
		"block_size": args.block_size,
		"n_layer": args.n_layer,
		"n_head": args.n_head,
		"n_embd": args.n_embd,
		"down_rate": args.down_rate,
		"use_external_embeddings": bool(args.use_external_embeddings),
		"epochs": int(args.epochs),
		"steps_per_epoch": int(steps_per_epoch),
		})
		csv_logger.close()

	# Clean up DDP
	if is_distributed:
		dist.barrier()
		dist.destroy_process_group()


if __name__ == "__main__":
	main()

