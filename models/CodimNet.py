'''
Zhang Z, Wei Y, Rao X, et al.
Electroencephalographic Biomarker-Guided Early Detection of Alzheimer’s Disease via Cortically Subdivided Neurodynamic PINN[J].
Expert Systems with Applications, 2025: 129353.
https://github.com/IMOP-lab/CodimNet
'''

import torch
import torch.nn as nn
from layers.CodimNet_EncDec import compute_ND_pinn_loss
from layers.CodimNet_EncDec import get_channel_indices, make_cnn_branch



class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Adapt kernel size based on input length to avoid size mismatch
        # Use min(100, seq_len//2) to ensure kernel_size doesn't exceed input
        self.first_kernel_size = min(100, max(32, self.seq_len // 4))
        self.second_kernel_size = min(22, max(11, self.seq_len // 8))

        # Get channel indices based on dataset
        self.lead_indices_list = get_channel_indices(configs.data if hasattr(configs, 'data') else 'ADFTD')

        self.lead_branches = nn.ModuleList([
            make_cnn_branch(len(self.lead_indices_list[0]), self.first_kernel_size, self.second_kernel_size),   # Frontal Lobe
            make_cnn_branch(len(self.lead_indices_list[1]), self.first_kernel_size, self.second_kernel_size),   # Central Lobe
            make_cnn_branch(len(self.lead_indices_list[2]), self.first_kernel_size, self.second_kernel_size),   # Parietal Lobe
            make_cnn_branch(len(self.lead_indices_list[3]), self.first_kernel_size, self.second_kernel_size),   # Occipital Lobe
            make_cnn_branch(len(self.lead_indices_list[4]), self.first_kernel_size, self.second_kernel_size),   # Temporal Lobe
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5) if not hasattr(configs, 'dropout') else nn.Dropout(configs.dropout)
        self.flatten = nn.Flatten()
        num_class = configs.num_class if hasattr(configs, 'num_class') else 3
        # Create two FC layers: one without age, one with age
        self.fc = nn.Linear(5 * 32, num_class)
        self.fc_with_age = nn.Linear(5 * 32 + 1, num_class)
        self.mid_features_list = [None] * 5
        self.original_x_list = [None] * 5

        for branch_idx in range(5):
            def create_hook(idx):
                def hook_fn(m, i, o):
                    self.mid_features_list[idx] = o
                return hook_fn
            target_layer = self.lead_branches[branch_idx][10]
            target_layer.register_forward_hook(create_hook(branch_idx))


    def supervised(self, x_enc, x_mark_enc, age=None):  # (batch_size, timestamps, enc_in)
        # Use dataset-specific channel indices (set in __init__)
        # Reshape if needed: (batch_size, timestamps, enc_in) -> (batch_size, enc_in, timestamps)
        if len(x_enc.shape) == 3:
            x = x_enc.transpose(1, 2)
        else:
            x = x_enc

        branch_outputs = []
        for idx, (indices, branch) in enumerate(zip(self.lead_indices_list, self.lead_branches)):
            input_x = x[:, indices, :]
            if idx < 5:
                self.original_x_list[idx] = input_x.detach()
            out = branch(input_x) 
            out = self.global_avg_pool(out) 
            branch_outputs.append(out)

        concat_output = torch.cat(branch_outputs, dim=1)

        x = self.dropout(concat_output)    
        x = self.flatten(x)                    
        if age is not None:
            x_with_age = torch.cat((x, age.reshape(-1, 1)), dim=1)   
            x = self.fc_with_age(x_with_age)
        else:
            x = self.fc(x)                          
        ND_pinn_losses = []
        for i in range(5):
            mid_feats = self.mid_features_list[i]
            original_feats = self.original_x_list[i]
            if mid_feats is not None:
                ND_loss_i = compute_ND_pinn_loss(
                    mid_feats, 
                    original_feats,
                    sample_rate_mid=100,   
                    sample_rate_orig=200,  
                )
                ND_pinn_losses.append(ND_loss_i)
            else:
                ND_pinn_losses.append(torch.tensor(0.0, device=x.device))

        ND_pinn_loss = torch.stack(ND_pinn_losses).mean()       
        return x, ND_pinn_loss


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, age=None):
        if self.task_name == 'supervised':
            dec_out, nd_loss = self.supervised(x_enc, x_mark_enc, age)
            return dec_out, nd_loss
        else:
            raise ValueError("Task name not recognized or not implemented within the CodimNet Model")


