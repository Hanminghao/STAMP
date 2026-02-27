import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class VisionPromptLearnerConch(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.trunk = base_model.visual.trunk
        self.pro_dim = base_model.visual.trunk.norm.weight.shape[0]
        self.dtype = base_model.visual.trunk.norm.weight.dtype
        self.attn_pool_contrast = base_model.visual.attn_pool_contrast
        self.ln_contrast = base_model.visual.ln_contrast
        self.p_visual = nn.Parameter(torch.empty(1, 1, self.pro_dim).type(self.dtype))
        self._global_pool = base_model.visual._global_pool
        self.ref_pos_embed = nn.Linear(self.pro_dim, self.pro_dim)

        for p in self.p_visual:
            nn.init.normal_(p, std=0.02)

    def forward(self, x, ref=None):
        x = x.type(self.dtype)
        x = self.trunk.patch_embed(x)
        x = self.trunk._pos_embed(x)

        if ref is not None:
            ref = self.trunk.patch_embed.proj(ref)
            ref = ref.flatten(2).transpose(1, 2)
            ref = self.trunk.patch_embed.norm(ref)
            ref = self.ref_pos_embed(ref)
            pretext_tokens = self.p_visual.expand(x.shape[0], -1, -1) * 0 + ref
        else:
            pretext_tokens = self.p_visual.expand(x.shape[0], -1, -1)

        x = torch.cat([x[:, 0].unsqueeze(1), pretext_tokens, x[:, 1:]], dim=1)
        hidden_states = self.trunk.norm_pre(x).type(self.dtype)

        for layer_idx, encoder_layer in enumerate(self.trunk.blocks):
            hidden_states = encoder_layer(hidden_states)

        cls_token, all_tokens = self._global_pool(hidden_states)
        if ref is not None:
            ref_token, all_tokens = all_tokens[:, 0], all_tokens[:, 1:]
            return cls_token, ref_token, all_tokens
        else:
            return cls_token, all_tokens
    

class VisionPromptLearnerUni(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.dtype = base_model.norm.weight.dtype
        self.p_visual = nn.Parameter(torch.empty(1, 1, base_model.embed_dim).type(self.dtype))
        self.ref_pos_embed = nn.Linear(base_model.embed_dim, base_model.embed_dim)
        self.base_model = base_model

        for p in self.p_visual:
            nn.init.normal_(p, std=0.02)

    def forward(self, x, ref=None):
        x = x.type(self.dtype)
        x = self.base_model.patch_embed(x)
        x = self.base_model._pos_embed(x)

        if ref is not None:
            ref = self.base_model.patch_embed.proj(ref)
            ref = ref.flatten(2).transpose(1, 2)
            ref = self.base_model.patch_embed.norm(ref)
            ref = self.ref_pos_embed(ref)
            pretext_tokens = self.p_visual.expand(x.shape[0], -1, -1) * 0 + ref
        else:
            pretext_tokens = self.p_visual.expand(x.shape[0], -1, -1)

        x = torch.cat([x[:, 0].unsqueeze(1), pretext_tokens, x[:, 1:]], dim=1)
        x = self.base_model.norm_pre(x).type(self.dtype)
        x = self.base_model.blocks(x)
        x = self.base_model.norm(x)
        if self.base_model.global_pool:
            cls_token = x[:, self.base_model.num_prefix_tokens:].mean(dim=1) if self.base_model.global_pool == 'avg' else x[:, 0]
        all_tokens = x[:, self.base_model.num_prefix_tokens:]
        if ref is not None:
            ref_token, all_tokens = all_tokens[:, 0], all_tokens[:, 1:]
            return cls_token, ref_token, all_tokens
        else:
            return cls_token, all_tokens
