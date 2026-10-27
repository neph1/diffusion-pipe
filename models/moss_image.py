import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageOps

from models.base import ComfyPipeline, make_contiguous, maybe_pad
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift
from utils.offloading import ModelOffloader
import comfy.latent_formats
from comfy.ldm.flux.layers import timestep_embedding
import comfy.ldm.moss_image.model
from comfy.ldm.moss_image.model import modulated_rms_norm, rope_split_half, situ, gated_residual
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention, ComfyAttention
import comfy.model_prefetch


def modulated_rms_norm_raw(x, norm):
    # training path of modulated_rms_norm without the scale/shift
    return norm(x)


def attention(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options, mask=None, isolate_split=None):
    # isolate_split: with edit references, the sequence is [live | refs]; reference
    # queries only attend to reference keys (ai-toolkit's kv_cache/isolate_refs mode).
    # Two unmasked SDPA calls when there is no padding; otherwise a block mask that
    # keeps reference queries on reference keys and lets live queries see everything.
    B, N, _ = q.shape
    dtype = v.dtype
    q = q.view(B, N, heads, -1)
    k = k.view(B, N, heads, -1)
    if comfy.model_management.in_training:
        q, k = rope_split_half(q_norm(q), pe).to(dtype), rope_split_half(k_norm(k), pe).to(dtype)
    else:
        q_scale, _, q_stream = comfy.ops.cast_bias_weight(q_norm, q, offloadable=True)
        k_scale, _, k_stream = comfy.ops.cast_bias_weight(k_norm, k, offloadable=True)
        q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, pe, q_scale, k_scale, q_norm.eps)
        comfy.ops.uncast_bias_weight(q_norm, q_scale, None, q_stream)
        comfy.ops.uncast_bias_weight(k_norm, k_scale, None, k_stream)
    if isolate_split is not None:
        # mask arrives as (B, 1, 1, N) key validity
        key_valid = mask.reshape(B, N)
        if bool(key_valid.all()):
            # no padding anywhere: isolation as two unmasked SDPA calls
            q_live, q_ref = q[:, :isolate_split], q[:, isolate_split:]
            k_live, k_ref = k[:, :isolate_split], k[:, isolate_split:]
            v_live, v_ref = v[:, :isolate_split], v[:, isolate_split:]
            out = torch.cat([
                optimized_attention(AttentionTensorContainer(q_live.flatten(2)), AttentionTensorContainer(k_live.flatten(2)), AttentionTensorContainer(v_live), heads, mask=None, transformer_options=transformer_options, preferred_attention=comfy_attention),
                optimized_attention(AttentionTensorContainer(q_ref.flatten(2)), AttentionTensorContainer(k_ref.flatten(2)), AttentionTensorContainer(v_ref), heads, mask=None, transformer_options=transformer_options, preferred_attention=comfy_attention),
            ], dim=1)
            return out
        # some sample has padding: build an explicit block mask instead
        is_ref = torch.arange(N, device=q.device) >= isolate_split
        allowed = key_valid[:, None, :] & (~is_ref[:, None] | is_ref[None, :])[None]
        allowed = allowed | torch.eye(N, dtype=torch.bool, device=q.device)
        mask = allowed[:, None]
    q, k, v = AttentionTensorContainer(q.flatten(2)), AttentionTensorContainer(k.flatten(2)), AttentionTensorContainer(v)
    return optimized_attention(q, k, v, heads, mask=mask, transformer_options=transformer_options, preferred_attention=comfy_attention)

# Support mask. Cast Q, K (float32) to V's dtype.
comfy.ldm.moss_image.model.attention = attention


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.norm1 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.qkv = operations.Linear(dim, 3 * dim, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.ffn_gate = operations.Linear(dim, ffn_dim, dtype=dtype, device=device)
        self.ffn_up = operations.Linear(dim, ffn_dim, dtype=dtype, device=device)
        self.ffn_down = operations.Linear(ffn_dim, dim, dtype=dtype, device=device)

    def forward(self, x, pe, mask, transformer_options={}):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        x = x + self.proj(attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads, self.comfy_attention, transformer_options, mask=mask))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))

comfy.ldm.moss_image.model.TextBlock = TextBlock


class TextEmbedder(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(text_dim, dim, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, num_heads, dtype=dtype, device=device, operations=operations) for _ in range(2)])

    def forward(self, x, pe, mask, transformer_options={}):
        x = self.proj(x)
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), x.device, transformer_options)
        for block in self.blocks:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, block, x.dtype)
            x = block(x, pe, mask, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, None)
        return x

comfy.ldm.moss_image.model.TextEmbedder = TextEmbedder


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, fused_qkv=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        self.norm1 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        # repacked files fuse q/k/v: one GEMM and one activation quantization instead of three
        names = ("qkv", "o") if fused_qkv else ("q", "k", "v", "o")
        self.proj = nn.ModuleDict({name: operations.Linear(dim, 3 * dim if name == "qkv" else dim, bias=False, dtype=dtype, device=device) for name in names})
        self.proj["ffn_up_gate"] = operations.Linear(dim, 2 * ffn_dim, bias=False, dtype=dtype, device=device)
        self.proj["ffn_down"] = operations.Linear(ffn_dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x, mod, pe, mask, transformer_options={}, isolate_split=None):
        # mod is (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f), each (B, 1, D),
        # or (mod, ref_mod, split) when edit references ride at t=0: tokens
        # [split:] are clean reference latents modulated at timestep 0
        if isinstance(mod, tuple):
            mod, ref_mod, split = mod
            m, r = mod, ref_mod

            def affine(h, scale, shift):
                return torch.cat(
                    (h[:, :split] * (1 + m[scale]) + m[shift],
                     h[:, split:] * (1 + r[scale]) + r[shift]), 1)

            def gate(h, g):
                return torch.cat((h[:, :split] * m[g], h[:, split:] * r[g]), 1)
        else:
            m = mod

            def affine(h, scale, shift):
                return h * (1 + m[scale]) + m[shift]

            def gate(h, g):
                return h * m[g]

        y = affine(modulated_rms_norm_raw(x, self.norm1), 1, 0)
        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, dim=-1)
        else:
            q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        out = attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads, self.comfy_attention, transformer_options, mask=mask, isolate_split=isolate_split)
        x = gated_residual(x, self.proj["o"](out), gate(m[2], 2))
        y = affine(modulated_rms_norm_raw(x, self.norm2), 4, 3)
        hidden, ffn_gate = self.proj["ffn_up_gate"](y).chunk(2, dim=-1)
        # situ without its 4 * 25 factor, applied on the gate instead (ffn_down has no bias): two fewer passes
        out = self.proj["ffn_down"](torch.tanh(ffn_gate / 4.0) * torch.sigmoid(ffn_gate) * torch.tanh(hidden / 25.0))
        return gated_residual(x, out, gate(m[5], 5) * 100.0)

comfy.ldm.moss_image.model.Block = Block


class MossImagePipeline(ComfyPipeline):
    name = 'moss_image'
    spatial_compression = 16
    channels = 128
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['Block', 'TextBlock']
    keep_in_high_precision = ['shared_block_modulation', 'text_embedder', 'time_embedding', 'video_embedder', 'final_linear_video', 'final_norm_video', 'rope.bands', 'video_head_modulation']

    # ai-toolkit edit layout: references are prepended to the prompt as
    # "Picture N: <vision>" and VLM-downscaled to <= vlm_image_size pixels
    vlm_image_size = 384 * 384
    # edit mode: references also ride the DiT as clean t=0 latents appended
    # after the text, isolated so their K/V are step-invariant (kv_cache)
    edit_mode = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_format = comfy.latent_formats.Flux2()
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.edit_mode = bool(self.model_config.get('edit_mode', False))

    def to_layers(self):
        diffusion_model = self.diffusion_model
        transformer_options = {
            'total_blocks': len(diffusion_model.layers),
            'block_type': 'single',
        }
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.layers):
            layers.append(TransformerLayer(block, i, transformer_options, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def patch_quantized_modules(self, model):
        num_patched_classes = super().patch_quantized_modules(model)
        if num_patched_classes > 0:
            raise NotImplementedError(f'Training LoRA on quantized weights will not work for MOSS-Image (training will not converge; training loss goes up sharply)')
        return 0

    def get_conds(self, inputs):
        text_embeds = inputs['text_embeds_0']
        attention_mask = inputs['attention_mask_0']
        # text embeds are variable length
        max_seq_len = max([e.size(0) for e in text_embeds])
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in text_embeds]
        )
        attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attention_mask]
        )
        assert text_embeds.shape[:2] == attention_mask.shape[:2]
        attention_mask = attention_mask.to(torch.bool)
        return text_embeds, attention_mask

    def get_call_vae_fn(self, vae):
        # edit mode: also encode the reference (control) image into control_latents
        parent_vae_fn = super().get_call_vae_fn(vae)

        def fn(*args):
            if len(args) == 1:
                return parent_vae_fn(args[0])
            elif len(args) == 2:
                tensor, control_tensor = args
                result = parent_vae_fn(tensor)
                control_result = parent_vae_fn(control_tensor)
                result['control_latents'] = control_result['latents']
                return result
            else:
                raise RuntimeError(f'Unexpected number of args: {len(args)}')
        return fn

    def load_image_for_vlm(self, path):
        pil_img = Image.open(path)
        height, width = pil_img.height, pil_img.width

        if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
            pil_img = pil_img.convert('RGBA')

        # add white background for transparent images
        if pil_img.mode == 'RGBA':
            canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
            canvas.alpha_composite(pil_img)
            pil_img = canvas.convert('RGB')
        else:
            pil_img = pil_img.convert('RGB')

        scale_factor = self.vlm_image_size / math.sqrt(height * width)
        return ImageOps.scale(pil_img, scale_factor)

    def get_call_text_encoder_fn(self, text_encoder):
        # 3-arg signature opts this text encoder into receiving control_file
        # (see DatasetManager.te_fn_requires_control_file)
        te = text_encoder
        tokenizer = getattr(te.tokenizer, te.tokenizer.clip)
        original_min_length = tokenizer.min_length

        def to_prompt(caption, images):
            # the ai-toolkit edit layout: "Picture N: <vision>" blocks ahead of
            # the prompt. MOSSImageTokenizer applies the chat template + think block.
            if len(images) == 0:
                return caption
            vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
            prefix = "".join(f"Picture {i + 1}: {vision_block}\n" for i in range(len(images)))
            return prefix + caption

        @torch.inference_mode()
        def fn(captions: list[str], is_video: list[bool], control_files: list[list[str]] | None):
            assert not any(is_video)

            images_per_caption = []
            for i in range(len(captions)):
                files = None if control_files is None else control_files[i]
                images_per_caption.append([self.load_image_for_vlm(f) for f in files] if files else [])

            # tokenize each caption separately: the vision blocks expand to
            # different token counts per image set
            all_token_lists = []
            for caption, images in zip(captions, images_per_caption):
                if len(images) == 0:
                    tokens = te.tokenize(caption)
                else:
                    image_tensors = torch.stack([
                        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                        for img in images
                    ])
                    tokens = te.tokenize(to_prompt(caption, images), images=[image_tensors[i:i + 1] for i in range(image_tensors.shape[0])])
                all_token_lists.append(tokens[te.tokenizer.clip_name][0])

            # no truncation: the image-pad runs must stay whole
            tokenizer.min_length = None
            max_length = max(len(t) for t in all_token_lists)
            tokens_dict = defaultdict(list)
            for token_list in all_token_lists:
                maybe_pad(token_list, max_length, tokenizer)  # right-pad to the longest in the batch
                tokens_dict[te.tokenizer.clip_name].append(token_list)

            o = te.encode_from_tokens_scheduled(tokens_dict)

            text_embeds = o[0][0].to(self.dtype)
            extra = o[0][1]
            if 'attention_mask' in extra:
                attention_mask = extra['attention_mask']
            else:
                attention_mask = torch.ones(text_embeds.shape[:2], dtype=torch.int64, device=text_embeds.device)

            tokenizer.min_length = original_min_length
            return {
                'text_embeds_0': text_embeds,
                'attention_mask_0': attention_mask,
            }

        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        bs, c, h, w = latents.shape
        device = latents.device

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        # edit mode: clean t=0 reference latents ride after the text tokens
        extra_inputs = ()
        if self.edit_mode and 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()
            extra_inputs = (control_latents,)

        return (noisy_latents, t, *self.get_conds(inputs)) + extra_inputs, (target, mask)

    def enable_block_swap(self, blocks_to_swap):
        diffusion_model = self.diffusion_model
        blocks = diffusion_model.layers
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        diffusion_model.layers = None
        diffusion_model.to('cuda')
        diffusion_model.layers = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.time_embeddings = model.time_embedding
        self.shared_block_modulation = model.shared_block_modulation
        self.video_head_modulation = model.video_head_modulation
        self.text_embedder = model.text_embedder
        self.video_embedder = model.video_embedder
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        if len(inputs) == 5:
            # edit mode: the last input carries the clean t=0 reference latents
            x, timestep, context, context_mask, ref_latents = inputs
        else:
            x, timestep, context, context_mask = inputs
            ref_latents = None

        B, _, H, W = x.shape
        dtype, device = x.dtype, x.device
        text_len = context.shape[1]
        pe = self.rope(H, W, text_len, device)

        vec = self.time_embedding(timestep_embedding(timestep, 256).to(dtype))
        mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
        head_shift, head_scale = self.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)

        text_mask = context_mask[:, None, None, :]
        txt = self.text_embedder(context, pe[:, H * W:], text_mask)
        # allocated outside the memory compiler region, the blocks update it in place
        x = torch.cat((self.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)
        del txt
        sizes = torch.tensor([H, W], dtype=torch.int32, device=device)

        img_len = H*W
        sequence_mask = torch.cat([
            torch.ones((B, 1, 1, img_len), dtype=bool, device=device),
            text_mask,
        ], dim=-1)

        isolate_split = None
        if ref_latents is not None:
            # clean reference latents (B, C, h_i, w_i) ride after the text at t=0.
            # ai-toolkit RoPE: reference i sits on axis r = i + 1 with the live
            # text length as its frame coordinate and its own centered grid.
            ref_tokens, ref_pe = self.pack_ref_latents(ref_latents, context_mask, device, dtype)
            x = torch.cat((x, self.video_embedder(ref_tokens)), dim=1)
            pe = torch.cat((pe.expand(B, -1, -1, -1, -1, -1), ref_pe), dim=1)
            ref_vec = self.time_embedding(timestep_embedding(torch.zeros_like(timestep), 256).to(dtype))
            ref_mod = self.shared_block_modulation(ref_vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
            mod = (mod, ref_mod, img_len + text_len)
            sequence_mask = torch.cat((sequence_mask, torch.ones((B, 1, 1, ref_tokens.shape[1]), dtype=bool, device=device)), dim=-1)
            isolate_split = img_len + text_len

        assert sequence_mask.shape[-1] == x.shape[1]

        return make_contiguous(x, pe, sequence_mask, head_shift, head_scale, sizes, isolate_split, *mod)

    def pack_ref_latents(self, ref_latents, context_mask, device, dtype):
        # (B, C, h, w) clean reference latents -> right-padded tokens (B, R, C)
        # and RoPE (B, R, head_dim); every sample carries exactly one reference.
        lengths = context_mask.sum(1)
        tokens = [latent.flatten(2).transpose(0, 1).to(device, dtype) for latent in ref_latents]
        num_ref = max(t.shape[0] for t in tokens)
        tokens = torch.stack([F.pad(t, (0, 0, 0, num_ref - t.shape[0])) for t in tokens])
        ids = torch.zeros(tokens.shape[0], num_ref, 4, device=device, dtype=torch.float32)
        for i, latent in enumerate(ref_latents):
            _, h, w = latent.shape
            ids[i, :h * w, 0] = i + 1
            ids[i, :h * w, 1] = lengths[i]
            ids[i, :h * w, 2] = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2).repeat_interleave(w)
            ids[i, :h * w, 3] = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2).repeat(h)
        pe = self.pe_embedder(ids).transpose(1, 2).contiguous()
        return tokens, pe.to(dtype)


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, transformer_options, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        transformer_options['block_index'] = block_idx
        self.transformer_options = transformer_options
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, pe, sequence_mask, head_shift, head_scale, sizes, isolate_split, *mod = inputs

        self.offloader.wait_for_block(self.block_idx)
        x = self.layer(x, mod, pe, sequence_mask, self.transformer_options, isolate_split=isolate_split)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, pe, sequence_mask, head_shift, head_scale, sizes, isolate_split, *mod)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_linear_video = model.final_linear_video
        self.final_norm_video = model.final_norm_video
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        x, pe, sequence_mask, head_shift, head_scale, sizes, isolate_split, *mod = inputs
        B = x.shape[0]
        H, W = sizes.tolist()
        out = self.final_linear_video(modulated_rms_norm(x[:, :H * W], self.final_norm_video, head_scale, head_shift))
        return out.transpose(1, 2).reshape(B, self.latent_channels, H, W)

