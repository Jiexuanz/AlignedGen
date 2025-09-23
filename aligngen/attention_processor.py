import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention
from dataclasses import dataclass
from diffusers.models.embeddings import apply_rotary_emb


@dataclass(frozen=True)
class StyleAlignedArgs:
    share_attention: bool = True
    block: tuple[int, int] = (19, 57)
    timesteps: tuple[int, int] = (0, 30)
    style_lambda_mode: str = "decrease"  # ["decrease", "fix"]
    style_lambda: float = 1.
    constrain_first: bool = True


T = torch.Tensor


def expand_first(feat: T, scale=1., ) -> T:
    bs = feat.shape[0]
    feat_style = feat[0].unsqueeze(0).repeat(bs, 1, 1, 1)
    return feat_style


def concat_first(feat: T, dim=2, scale=1.) -> T:
    bs = feat.shape[0]
    feat_style = feat[0].unsqueeze(0).repeat(bs, 1, 1, 1)
    return torch.cat((feat, feat_style), dim=dim)


def concat_first_block(feat_all: T, feat_block: T, dim=2, scale=1.) -> T:
    bs = feat_all.shape[0]
    if scale == 1.:
        feat_style = feat_block[0].unsqueeze(0).repeat(bs, 1, 1, 1)
    else:
        feat_style = (scale * feat_block[0]).unsqueeze(0).repeat(bs - 1, 1, 1, 1)
        feat_style = torch.cat((feat_block[0].unsqueeze(0), feat_style), dim=0)
    return torch.cat((feat_all, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


class ShareAttnFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, cnt: int, style_aligned_args: StyleAlignedArgs):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.cnt = cnt
        self.t = 1
        self.args = style_aligned_args
        self.attn_weights = []

    def set_timesteps(self, t, timesteps):
        self.t = t
        if self.args.style_lambda_mode == "decrease":
            self.scale = (timesteps / 1000) * (timesteps / 1000)
        elif self.args.style_lambda_mode == "fix":
            self.scale = self.args.style_lambda

    def set_args(self, style_aligned_args):
        self.args = style_aligned_args

    def ori_call(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            image_rotary_emb_additional: Optional[torch.Tensor] = None,
            txt_length: int = None,
    ) -> torch.FloatTensor:
        if not self.args.share_attention:
            return self.ori_call(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )
        if not ((self.cnt >= self.args.block[0] and self.cnt < self.args.block[1])
                and (self.t >= self.args.timesteps[0] and self.t < self.args.timesteps[1])):
            return self.ori_call(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # *****************************************************************************
        # crop hidden_states into -> encoder_hidden_states & image hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states_q = query[:, :, :txt_length, :]
            encoder_hidden_states_k = key[:, :, :txt_length, :]
            encoder_hidden_states_v = value[:, :, :txt_length, :]
            query = query[:, :, txt_length:, :]
            key = key[:, :, txt_length:, :]
            value = value[:, :, txt_length:, :]
        # *****************************************************************************

        query = adain(query)
        key = adain(key)

        # *****************************************************************************
        if encoder_hidden_states is None:
            query = torch.cat([encoder_hidden_states_q, query], dim=-2)
            key = torch.cat([encoder_hidden_states_k, key], dim=-2)
            value = torch.cat([encoder_hidden_states_v, value], dim=-2)
        # *****************************************************************************

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # *****************************************************************************
        if txt_length is None:
            assert encoder_hidden_states is not None
            txt_length = encoder_hidden_states.shape[1]

        k_ = key[:, :, txt_length:, :]
        v_ = value[:, :, txt_length:, :]
        k_ = apply_rotary_emb(k_, image_rotary_emb_additional)
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)
        key = concat_first_block(key, k_, -2, scale=self.scale)
        value = concat_first_block(value, v_, -2)
        # *****************************************************************************

        # *****************************************************************************
        if self.args.constrain_first:
            rows, cols = query.shape[-2], key.shape[-2]
            attn_mask = torch.zeros((query.shape[0], 24, rows, cols), dtype=query.dtype, device=query.device)
            attn_mask[0, :, :, query.shape[-2]:] = -float("inf")
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False,
                                                           attn_mask=attn_mask)
            del attn_mask
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        # *****************************************************************************

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
