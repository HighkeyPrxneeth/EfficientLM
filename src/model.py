import math
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from torch.utils.checkpoint import checkpoint

class SwiGLU(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: Optional[int] = None):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else (in_dim * 8) // 3
        self.w1 = nn.Linear(in_dim, self.hidden_dim)
        self.w2 = nn.Linear(in_dim, self.hidden_dim)
        self.w3 = nn.Linear(self.hidden_dim, in_dim)
    
    def forward(self, x):
        gate = self.w1(x)
        data = self.w2(x)
        out = self.w3(F.silu(gate) * data)
        return out

class SwiGLUAct(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b

class Adapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bottleneck: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, bottleneck, bias=False)
        self.w2 = nn.Linear(in_dim, bottleneck, bias=False)
        self.up_proj = nn.Linear(bottleneck, out_dim, bias=False)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        self.ln = nn.LayerNorm(out_dim)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        h = F.silu(self.w1(x)) * self.w2(x)
        h = self.up_proj(h)
        out = self.ln(h + self.skip(x))
        return out


class HyperNetwork(nn.Module):
    def __init__(self, d, down_rate, low_rank=4):
        super().__init__()
        self.d = d
        self.down_rate = down_rate
        self.fc_c = nn.Linear(d, int(d / low_rank))
        self.fc_p = nn.Linear(d, int(d / low_rank))

    def positional_encoding(self, T, pos=0):
        d = self.d
        position = torch.arange(pos, pos + T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d))
        pe = torch.zeros(T, d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, T, t, device, train=True, low_rank=False, T_input=None, t_input=None):
        if not train and T_input is not None and T_input.shape[1] == 1:
            P = T_input  # [B, 1, d]
            C = t_input if t_input is not None else self.positional_encoding(1, pos=t - 1).to(device)  # [1, 1, d]
            # Expand C to batch size of P
            if C.shape[0] != P.shape[0]:
                C = C.expand(P.shape[0], -1, -1)
            C2 = self.fc_c(C)
            P2 = self.fc_p(P)
            W = torch.bmm(C2, P2.transpose(1, 2))
            return torch.sigmoid(W)

        P = T_input if T_input is not None else self.positional_encoding(T).to(device)  # [B or 1, T, d]
        if t_input is not None:
            C = t_input
        else:
            C = self.positional_encoding(t).to(device).to(P.dtype)
            if train:
                C = C.repeat_interleave(self.down_rate, dim=1)[:, :T]
        # Expand C to match batch size of P
        if C.shape[0] != P.shape[0]:
            C = C.expand(P.shape[0], -1, -1)
        C2 = self.fc_c(C)
        P2 = self.fc_p(P)
        W = torch.bmm(C2, P2.transpose(1, 2))
        return torch.sigmoid(W)


class MultiheadTemporalLatentAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        q_lora_rank: int = 0,
        kv_lora_rank: int = 256,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        down_rate: int = 2,
        recompute_prompt_attn: bool = False,
        rope_base: float = 10000.0,
        rope_scale: float = 1.0,
        decouple_norm: bool = True,
        learnable_pos_scale: bool = True,
        init_pos_scale: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.down_rate = down_rate
        self.recompute_prompt_attn = recompute_prompt_attn
        self.rope_base = rope_base
        self.rope_scale = rope_scale
        self.decouple_norm = decouple_norm
        self.pos_scale = nn.Parameter(torch.tensor(init_pos_scale)) if learnable_pos_scale else None

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(embed_dim, num_heads * self.qk_head_dim, bias=bias)
        else:
            self.wq_a = nn.Linear(embed_dim, q_lora_rank, bias=bias)
            self.q_norm = nn.LayerNorm(q_lora_rank)
            self.wq_b = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=bias)

        self.wkv_a = nn.Linear(embed_dim, kv_lora_rank + qk_rope_head_dim, bias=bias)
        self.kv_norm = nn.LayerNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=bias)
        self.wo = nn.Linear(num_heads * v_head_dim, embed_dim, bias=bias)
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.hypernet_down = HyperNetwork(d=kv_lora_rank, down_rate=down_rate)
        self.init_incremental_state()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        need_weights: bool = False,
    ):
        bsz, seqlen, _ = query.size()
        if self.q_lora_rank == 0:
            q = self.wq(query)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(query)))
        q = q.view(bsz, seqlen, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            start_pos = saved_state.get("infer_steps", None)[0] if "infer_steps" in saved_state else key.shape[1] - 1
        else:
            start_pos = 0

        if position is None:
            position = torch.arange(start_pos, start_pos + seqlen, device=query.device).unsqueeze(0).expand(bsz, -1)

        sin, cos = self._compute_sin_cos_batch(position, query.device)
        q_pe = self._apply_rotary_emb_sin_cos(q_pe, sin[:, -seqlen:], cos[:, -seqlen:])

        kv = self.wkv_a(key)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = self._apply_rotary_emb_sin_cos(k_pe.unsqueeze(2), sin, cos)
        kv_norm = self.kv_norm(kv)
        k_pe = k_pe.squeeze(2)

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            prev_kv_t = saved_state.get("prev_kv_t", None)
            prev_k_pe = saved_state.get("prev_k_pe", None)
            infer_steps = saved_state.get("infer_steps", None)

            T = start_pos + 1
            t = math.ceil(T / self.down_rate)
            T_remain = T % self.down_rate

            w_tT = self.hypernet_down(T, t, kv_norm.device, train=False, T_input=kv_norm)

            tricky_mask = None
            if prev_kv_t is not None and prev_k_pe is not None:
                if T_remain != 1:
                    prev_kv_t[:, -1:] += kv_norm * w_tT
                    prev_k_pe[:, -1:] = k_pe
                else:
                    prev_kv_t = torch.cat([prev_kv_t, kv_norm * w_tT], dim=1)
                    prev_k_pe = torch.cat([prev_k_pe, k_pe], dim=1)

                saved_state["prev_kv_t"] = prev_kv_t
                saved_state["prev_k_pe"] = prev_k_pe
                infer_steps = infer_steps + 1
            else:
                if key.shape[1] != 1:
                    indices = list(range(self.down_rate - 1, T, self.down_rate))
                    if T - 1 not in indices:
                        indices.append(T - 1)
                    if self.recompute_prompt_attn:
                        w_tT = self.hypernet_down(T, t, kv_norm.device, train=True, T_input=kv_norm)
                        zero_mask = self.generate_chunk_mask(T, self.down_rate).to(k_pe.device).unsqueeze(0).to(kv_norm.dtype)
                        prev_kv_t = torch.matmul(w_tT * zero_mask, kv_norm)
                        prev_k_pe = k_pe
                        saved_state["prev_kv_t"] = prev_kv_t[:, indices]
                        saved_state["prev_k_pe"] = prev_k_pe[:, indices]
                        tricky_mask = self.generate_stride_aware_causal_mask(T).to(prev_kv_t.device)
                        if seqlen != T:
                            tricky_mask = tricky_mask[-seqlen:]
                    else:
                        zero_mask = self.generate_chunk_mask(T, self.down_rate).to(k_pe.device)
                        indices = list(range(self.down_rate - 1, T, self.down_rate))
                        if T - 1 not in indices:
                            indices.append(T - 1)
                        zero_mask = zero_mask[indices].unsqueeze(0)
                        prev_kv_t = torch.matmul(w_tT * zero_mask, kv_norm)
                        prev_k_pe = k_pe[:, indices]
                        saved_state["prev_kv_t"] = prev_kv_t
                        saved_state["prev_k_pe"] = prev_k_pe
                else:
                    prev_kv_t = kv_norm * w_tT
                    prev_k_pe = k_pe
                    saved_state["prev_kv_t"] = prev_kv_t
                    saved_state["prev_k_pe"] = prev_k_pe
                infer_steps = kv_norm.new_zeros(kv_norm.shape[0]) + T

            saved_state["infer_steps"] = infer_steps
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

            wkv_b = self.wkv_b.weight.view(self.num_heads, -1, self.kv_lora_rank)
            q_nope_proj = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim])
            content_scores = torch.einsum("bshc,btc->bsht", q_nope_proj, prev_kv_t)
            if self.decouple_norm:
                q_pe_n = F.normalize(q_pe, dim=-1)
                k_pe_n = F.normalize(prev_k_pe, dim=-1)
                pos_scores = torch.einsum("bshr,btr->bsht", q_pe_n, k_pe_n)
            else:
                pos_scores = torch.einsum("bshr,btr->bsht", q_pe, prev_k_pe)
            pos_scale = self.pos_scale if self.pos_scale is not None else 1.0
            scores = (content_scores + pos_scale * pos_scores) * self.softmax_scale
            if tricky_mask is not None:
                scores = scores + tricky_mask.unsqueeze(0).unsqueeze(2).to(scores.dtype)
            if self_attn_mask is not None:
                scores = scores + self_attn_mask.unsqueeze(0).unsqueeze(2)
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            x = torch.einsum("bsht,btc->bshc", attn_weights, prev_kv_t)
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
            x = self.wo(x.flatten(2))
            return x
        else:
            T = key.size(1)
            t = math.ceil(T / self.down_rate)
            w_tT = self.hypernet_down(T, t, kv_norm.device, train=True, T_input=kv_norm)
            zero_mask = self.generate_chunk_mask(T, self.down_rate).to(k_pe.device).unsqueeze(0).to(kv_norm.dtype)
            kv_norm_t_full = torch.matmul(w_tT * zero_mask, kv_norm)  # [B, T, C]
            wkv_b = self.wkv_b.weight.view(self.num_heads, -1, self.kv_lora_rank)
            q_nope_proj = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim])
            # Select stride-aligned key positions to keep last in each chunk; build mask of shape [S, t]
            indices = list(range(self.down_rate - 1, T, self.down_rate))
            if T - 1 not in indices:
                indices.append(T - 1)
            # Reduce aggregated keys to stride-aligned set so both content and positional paths share same cols
            kv_norm_t = kv_norm_t_full[:, indices]  # [B, t, C]
            # Base causal mask [T, T], then select columns at stride indices
            base_mask = torch.full((T, T), -1e9, device=q_nope_proj.device)
            rows = torch.arange(T, device=q_nope_proj.device).view(-1, 1)
            cols = torch.arange(T, device=q_nope_proj.device).view(1, -1)
            base_mask[(cols <= rows)] = 0
            tricky_mask = base_mask[:, indices]
            if seqlen != T:
                tricky_mask = tricky_mask[-seqlen:]
            content_scores = torch.einsum("bshc,btc->bsht", q_nope_proj, kv_norm_t)
            if self.decouple_norm:
                q_pe_n = F.normalize(q_pe, dim=-1)
                k_pe_ds = k_pe[:, indices]
                k_pe_n = F.normalize(k_pe_ds, dim=-1)
                pos_scores = torch.einsum("bshr,btr->bsht", q_pe_n, k_pe_n)
            else:
                k_pe_ds = k_pe[:, indices]
                pos_scores = torch.einsum("bshr,btr->bsht", q_pe, k_pe_ds)
            pos_scale = self.pos_scale if self.pos_scale is not None else 1.0
            scores = (content_scores + pos_scale * pos_scores) * self.softmax_scale
            scores = scores + tricky_mask.unsqueeze(0).unsqueeze(2).to(scores.dtype)
            if self_attn_mask is not None:
                scores = scores + self_attn_mask.unsqueeze(0).unsqueeze(2)
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            x = torch.einsum("bsht,btc->bshc", attn_weights, kv_norm_t)
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
            x = self.wo(x.flatten(2))
            return x

    def _compute_sin_cos_batch(self, pos: torch.Tensor, device: torch.device):
        """Compute RoPE angles' sin and cos without complex numbers.

        Returns:
            sin: [B, T, D/2]
            cos: [B, T, D/2]
        """
        theta = self.rope_base
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.qk_rope_head_dim, 2, device=device, dtype=torch.float32) / self.qk_rope_head_dim))
        if self.rope_scale != 1.0:
            inv_freq = inv_freq / self.rope_scale
        angles = torch.einsum("bi,j->bij", pos.to(dtype=torch.float32), inv_freq)  # [B, T, D/2]
        return torch.sin(angles), torch.cos(angles)

    def _apply_rotary_emb_sin_cos(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """Apply RoPE given precomputed sin and cos. x: [B, T, H, D]"""
        dtype = x.dtype
        x_float = x.float().view(*x.shape[:-1], -1, 2)
        x1 = x_float[..., 0]
        x2 = x_float[..., 1]
        # Broadcast sin/cos over heads
        sin = sin.unsqueeze(2)
        cos = cos.unsqueeze(2)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        y = torch.stack((y1, y2), dim=-1).flatten(-2)
        return y.to(dtype)

    def generate_chunk_mask(self, T, chunk_size):
        row_indices = torch.arange(T).view(-1, 1)
        col_indices = torch.arange(T).view(1, -1)
        row_chunk = row_indices // chunk_size
        col_chunk = col_indices // chunk_size
        same_chunk = row_chunk == col_chunk
        tril_mask = row_indices % chunk_size >= col_indices % chunk_size
        chunk_mask = same_chunk & tril_mask
        return chunk_mask.float()

    def generate_stride_aware_causal_mask(self, T):
        mask = torch.full((T, T), -1e9)
        rows = torch.arange(T).view(-1, 1)
        cols = torch.arange(T).view(1, -1)
        visible = ((cols <= rows) & ((cols + 1) % self.down_rate == 0)) | (cols == rows)
        mask[visible] = 0
        return mask

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return f"{self._incremental_state_id}.{key}"

    def get_incremental_state(self, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]], key: str):
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(self, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]], key: str, value: Dict[str, Optional[torch.Tensor]]):
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]):
        result = self.get_incremental_state(incremental_state, "attn_state")
        return result if result is not None else {}

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]], buffer: Dict[str, Optional[torch.Tensor]]):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def reorder_incremental_state(self, incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]], new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in list(input_buffer.keys()):
                if input_buffer[k] is not None:
                    input_buffer[k] = input_buffer[k].index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state


class TransformerBlockMTLA(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        kv_lora_rank: int = 256,
        down_rate: int = 2,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
    v_head_dim: Optional[int] = None,
    gradient_checkpoint: bool = False,
    ):
        super().__init__()
        if v_head_dim is None:
            v_head_dim = n_embd // n_head
        if qk_nope_head_dim is None:
            qk_nope_head_dim = v_head_dim
        if qk_rope_head_dim is None:
            qk_rope_head_dim = max(1, v_head_dim // 2)
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiheadTemporalLatentAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=attn_dropout,
            bias=False,
            q_lora_rank=0,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            down_rate=down_rate,
            recompute_prompt_attn=False,
    )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(SwiGLU(in_dim=n_embd), nn.Dropout(mlp_dropout))
        self.resid_drop = nn.Dropout(resid_dropout)
        self.gradient_checkpoint = gradient_checkpoint

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
    ) -> torch.Tensor:
        def _forward_impl(x_in: torch.Tensor, kpm: Optional[torch.Tensor], pos: Optional[torch.Tensor], sam: Optional[torch.Tensor]) -> torch.Tensor:
            h = self.ln1(x_in)
            attn_out = self.attn(
                h,
                h,
                h,
                key_padding_mask=kpm,
                self_attn_mask=sam,
                position=pos,
                incremental_state=incremental_state,
                need_weights=False,
            )
            y = x_in + self.resid_drop(attn_out)
            y = y + self.resid_drop(self.mlp(self.ln2(y)))
            return y

        if self.gradient_checkpoint and self.training and incremental_state is None:
            return checkpoint(_forward_impl, x, key_padding_mask, position, self_attn_mask)
        else:
            return _forward_impl(x, key_padding_mask, position, self_attn_mask)


class LMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        use_external_embeddings: bool = True,
        ext_emb_dim: int = 768,
        adapter_bottleneck: int = 64,
        dropout: float = 0.0,
    kv_lora_rank: int = 256,
    down_rate: int = 2,
    gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_external_embeddings = use_external_embeddings
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.adapter = Adapter(in_dim=ext_emb_dim, out_dim=n_embd, bottleneck=adapter_bottleneck)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockMTLA(
                    n_embd=n_embd,
                    n_head=n_head,
                    attn_dropout=dropout,
                    resid_dropout=dropout,
                    mlp_dropout=dropout,
                    kv_lora_rank=kv_lora_rank,
                    down_rate=down_rate,
                    gradient_checkpoint=gradient_checkpoint,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        x_tokens: Optional[torch.Tensor] = None,
        x_external: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        incremental_states: Optional[List[Dict[str, Dict[str, Optional[torch.Tensor]]]]] = None,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x_tokens: [B, T] token ids when using internal embeddings.
            x_external: [B, T, ext_emb_dim] external embeddings when adapter path is enabled.
            targets: [B, T] token ids for loss.
            key_padding_mask: [B, T] boolean mask where True marks padding to be ignored in attention.
            self_attn_mask: [T, T] additive mask for attention scores.
            incremental_states: list of per-block dicts for KV caching; pass the same list across timesteps for generation.
            ignore_index: target id to ignore in loss computation.
        Returns:
            logits: [B, T, vocab]
            loss: scalar or None
        """
        B = T = None  # type: ignore
        h = None
        if x_tokens is not None:
            h = self.tok_emb(x_tokens)
            B, T = x_tokens.shape
        if x_external is not None:
            cond = self.adapter(x_external)
            if h is None:
                h = cond
                B, T, _ = h.shape
            else:
                h = h + cond
        if h is None or B is None or T is None:
            raise ValueError("Provide at least one of x_tokens or x_external to LMModel.forward")

        # Enforce maximum block size for positions
        if T > self.pos_emb.num_embeddings:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.pos_emb.num_embeddings}")

        # Build positional encodings and masks
        pos = torch.arange(T, device=h.device, dtype=torch.long)
        # Add learned positional embedding except when doing single-token incremental step
        is_inc = incremental_states is not None and len(incremental_states) == len(self.blocks)
        if not (is_inc and T == 1):
            h = h + self.pos_emb(pos)[None, :, :]
        h = self.drop(h)
        # Pass absolute positions during training/full prompt; let attention infer during 1-token incremental
        position = None if (is_inc and T == 1) else pos.unsqueeze(0).expand(B, -1)

        kpm = None
        if key_padding_mask is not None:
            kpm = key_padding_mask.to(torch.bool).to(h.device)

        # Prepare incremental states per block if provided
        use_inc = incremental_states is not None and len(incremental_states) == len(self.blocks)

        for i, blk in enumerate(self.blocks):
            inc = incremental_states[i] if use_inc else None
            h = blk(
                h,
                key_padding_mask=kpm,
                position=position,
                self_attn_mask=self_attn_mask,
                incremental_state=inc,
            )
        h = self.ln_f(h)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV caching (token embedding path only).

        Args:
            idx: [B, T] prompt token ids.
        """
        if self.use_external_embeddings:
            raise NotImplementedError("generate() currently supports only token-embedding path (use_external_embeddings=False)")

        def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
            if k is None or k <= 0:
                return logits
            v, _ = torch.topk(logits, k)
            min_keep = v[..., -1, None]
            return torch.where(logits < min_keep, torch.full_like(logits, -1e10), logits)

        self.eval()
        B, T = idx.shape
        device = idx.device
        # Initialize per-block incremental states
        inc_states: List[Dict[str, Dict[str, Optional[torch.Tensor]]]] = [dict() for _ in self.blocks]

        # Prime caches with the full prompt (may be truncated to block size for positions)
        idx_cond = idx[:, -self.pos_emb.num_embeddings :]
        logits, _ = self.forward(x_tokens=idx_cond, incremental_states=inc_states)

        for _ in range(max_new_tokens):
            # Take last token's logits
            last_logits = logits[:, -1, :] / max(temperature, 1e-8)
            last_logits = top_k_filter(last_logits, top_k)
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if eos_id is not None:
                # If all batches ended, early stop (simple check without masking per batch)
                if torch.all(next_token.squeeze(-1) == eos_id):
                    break

            # Feed only the new token to advance caches
            logits, _ = self.forward(x_tokens=next_token, incremental_states=inc_states)

        return idx