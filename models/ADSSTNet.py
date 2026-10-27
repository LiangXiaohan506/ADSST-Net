import torch
import torch.nn as nn
from layers.ADSSTNet_EncDec import Encoder



class _PretrainingClaHead(nn.Module):
    """对每个时间步的 d_model 做投影；BatchNorm 作用在通道维 [B, L, C]。"""

    def __init__(self, d_model: int, projection_dim: int, dropout_head: float, num_class: int):
        super().__init__()
        self.lin1 = nn.Linear(in_features=d_model, out_features=projection_dim, bias=True)

    def forward(self, x):
        # x: [B, L, C]
        x = self.lin1(x)
        return x


#%%
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name

        A0_tensor = torch.tensor(configs.A0, dtype=torch.float32) if configs.A0 is not None else None
        self.encoder = Encoder(
            configs.e_layers, configs.d_model, configs.seq_len, configs.embedding_dim,
            configs.d_state, configs.d_conv_in, configs.d_conv, configs.d_conv_out, configs.dt_rank,
            configs.expand, configs.bimamba, configs.space, configs.se_reduction,
            dim_UVA=configs.dim_UVA, A0=A0_tensor, topk_A0=configs.topk_A0,
            sample_adapt_A0=configs.sample_adapt_A0, sample_adapt_Fre=configs.sample_adapt_Fre,
            num_heads=configs.num_heads,
            downsample_min_len=getattr(configs, "downsample_min_len", None),
            dataset_name=getattr(configs, "data", None),
            preserve_frontal_occipital=getattr(configs, "preserve_frontal_occipital", False),
        )

        if self.task_name in ("supervised"):
            self.adaptiveAvgPool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.projection_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(configs.dropout_head),
                nn.Linear(in_features=configs.projection_dim, out_features=configs.num_class, bias=True)
            )
            self.projection_heads = nn.ModuleList(
                _PretrainingClaHead(configs.d_model, configs.projection_dim, configs.dropout_head, configs.num_class)
                for _ in range(self.encoder.num_layers)
            )


    def supervised(self, x_enc, x_mark_enc=None, model_state=None):
        enc_out, ASM_visualization_list, PGL_visualization_list, RGDI_visualization_list = self.encoder(x_enc)
        # enc_out = self.encoder(x_enc)
        if model_state != 'test':
            layer_out = tuple(h(self.adaptiveAvgPool(z.permute(0, 2, 1)).squeeze(-1)) for h, z in zip(self.projection_heads, enc_out))
            # layer_out = self.projection_heads(self.adaptiveAvgPool(enc_out[-1].permute(0, 2, 1)).squeeze(-1))
            # layer_out = tuple(self.classifier(self.adaptiveAvgPool(z.permute(0, 2, 1)).squeeze(-1)) for z in enc_out[:-1])
            # layer_out = tuple(h(z.permute(0, 2, 1)).squeeze(-1) for h, z in zip(self.projection_heads, enc_out))
            # layer_out= enc_out[:-1]
        else:
            layer_out = enc_out[:-1]
        logits = self.classifier(self.adaptiveAvgPool(enc_out[-1].permute(0, 2, 1)).squeeze(-1))
        return layer_out, logits, enc_out, ASM_visualization_list, PGL_visualization_list, RGDI_visualization_list


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, model_state=None):
        if self.task_name == "supervised":
            return self.supervised(x_enc, x_mark_enc, model_state)
        else:
            raise ValueError("Task name not recognized or not implemented within the ADMamba model")

