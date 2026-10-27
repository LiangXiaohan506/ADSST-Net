# @inproceedings{tslanet,
#   title     = {TSLANet: Rethinking Transformers for Time Series Representation Learning},
#   author    = {Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Li, Xiaoli},
#   booktitle = {International Conference on Machine Learning},
#   year      = {2024}
# }

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from layers.TSLANet_EncDec import PatchEmbed, TSLANet_layer
from utils.masking import random_masking_3D


class Model(nn.Module):
    """
    TSLANet: Rethinking Transformers for Time Series Representation Learning
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.patch_embed = PatchEmbed(
            seq_len=configs.seq_len,
            patch_size=configs.patch_size,
            in_chans=configs.enc_in,
            embed_dim=configs.embedding_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, configs.embedding_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=configs.dropout)

        self.input_layer = nn.Linear(configs.patch_size, configs.embedding_dim)

        dpr = [x.item() for x in torch.linspace(0, configs.dropout, configs.e_layers)]

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(
                dim=configs.embedding_dim,
                use_icb=configs.use_icb,
                use_asb=configs.use_asb,
                drop=configs.dropout,
                drop_path=dpr[i],
                adaptive_filter=configs.adaptive_filter
            )
            for i in range(configs.e_layers)
        ])

        if self.task_name == 'supervised':
            self.head = nn.Linear(configs.embedding_dim, configs.num_class)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pretrain(self, x_in, masking_ratio):
        x = self.patch_embed(x_in)
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)

        x_masked, _, self.mask, _ = random_masking_3D(x, mask_ratio=masking_ratio)
        self.mask = self.mask.bool()  # mask: [bs x num_patch x n_vars]

        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)

        return x_masked, x_patched

    def supervised(self, x_enc, x_mark_enc):
        # x_enc: (batch_size, seq_len, enc_in) -> transpose to (batch_size, enc_in, seq_len) for Conv1d
        x = x_enc.transpose(1, 2)
        x = self.patch_embed(x) # [batch_size, patch_size, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        x = x.mean(1)
        x = self.head(x)
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'supervised':
            dec_out = self.supervised(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError("Task name not recognized or not implemented within the TSLANet Model")
