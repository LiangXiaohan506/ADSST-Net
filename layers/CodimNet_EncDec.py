'''
Zhang Z, Wei Y, Rao X, et al.
Electroencephalographic Biomarker-Guided Early Detection of Alzheimer’s Disease via Cortically Subdivided Neurodynamic PINN[J].
Expert Systems with Applications, 2025: 129353.
https://github.com/IMOP-lab/CodimNet
'''

import torch
import torch.nn as nn



# Define the frequency range of the band (unit: Hz)
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}


def compute_band_power(signal, sample_rate):
    fft = torch.fft.rfft(signal, dim=-1)
    power_spectrum = torch.abs(fft) ** 2
    freqs = torch.fft.rfftfreq(signal.size(-1), d=1. / sample_rate).to(signal.device)

    band_powers = []
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        fmin, fmax = FREQ_BANDS[band]
        idx = (freqs >= fmin) & (freqs < fmax)
        band_power = power_spectrum[..., idx].mean(dim=-1)
        band_powers.append(band_power)

    band_powers = torch.stack(band_powers, dim=-1)
    band_powers = band_powers.mean(dim=1)
    return band_powers


def compute_ND_pinn_loss(mid_features, original_signal,
                         sample_rate_mid=100, sample_rate_orig=200):
    mid_band_power = compute_band_power(mid_features, sample_rate_mid)
    original_band_power = compute_band_power(original_signal, sample_rate_orig)

    relative_diff = (mid_band_power - original_band_power) / (original_band_power + 1e-6)

    per_band_mse = (relative_diff ** 2).mean(dim=0)
    weights = 1.0 / (per_band_mse + 1e-6)
    weights = weights / weights.sum()

    ND_pinn_loss = (weights * per_band_mse).sum()
    return ND_pinn_loss


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = self._make_layer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(growth_rate),
            nn.ReLU()
        )
        return layer

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x


def get_channel_indices(dataset_name):
    """
    Get channel indices for different brain regions based on dataset.

    Args:
        dataset_name: Name of the dataset ('ADFTD', 'CAUEEG', 'USTBMCI')

    Returns:
        List of channel indices for [Frontal, Central, Parietal, Occipital, Temporal] lobes
    """
    # Define channel orders for each dataset
    channel_orders = {
        'ADFTD': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4',
                  'T6', 'O1', 'O2'],
        'CAUEEG': ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F7', 'T3', 'T5', 'F8', 'T4', 'T6',
                   'FZ', 'CZ', 'PZ'],
        'CAEEG': ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'T7', 'T8', 'Pz', 'P3', 'P4',
                    'P7', 'P8', 'Oz', 'O1', 'O2'],
        'IUMSMCI': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4',
                  'T6', 'O1', 'O2'],
    }

    # Define brain regions (normalize channel names to handle case variations and T3/T7 equivalents)
    brain_regions = {
        'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'FZ'],
        'Central': ['Cz', 'CZ', 'C3', 'C4'],
        'Parietal': ['Pz', 'PZ', 'P3', 'P4'],
        'Occipital': ['O1', 'O2', 'Oz', 'OZ'],
        'Temporal': ['F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'P7', 'P8']
    }

    # Get channel order for the specified dataset
    if dataset_name not in channel_orders:
        print(f"Warning: Unknown dataset '{dataset_name}', using ADFTD as default")
        dataset_name = 'ADFTD'

    channel_order = channel_orders[dataset_name]

    # Find indices for each brain region
    indices_list = []
    for region_name in ['Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal']:
        region_channels = brain_regions[region_name]
        indices = []
        for ch in region_channels:
            if ch in channel_order:
                indices.append(channel_order.index(ch))

        if len(indices) == 0:
            print(f"Warning: No channels found for {region_name} region in {dataset_name} dataset")

        indices_list.append(indices)

    return indices_list


def make_cnn_branch(in_channels, first_kernel_size, second_kernel_size):
    layers = []
    # First conv: use adaptive kernel size based on input length
    first_kernel = first_kernel_size
    first_padding = (first_kernel - 1) // 2  # 'same' padding equivalent for stride=2
    layers.append(nn.Conv1d(in_channels, 32, kernel_size=first_kernel, stride=2, padding=first_padding))
    layers.append(nn.BatchNorm1d(32))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool1d(kernel_size=2, stride=1, padding=0))

    # Depthwise separable conv: use adaptive kernel size
    second_kernel = second_kernel_size
    second_padding = (second_kernel - 1) // 2  # 'same' padding for stride=1
    layers.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=second_kernel, stride=1, groups=32,
                            padding=second_padding))
    layers.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1))
    layers.append(nn.BatchNorm1d(32))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool1d(kernel_size=2, stride=1, padding=0))

    # Dense Block 1
    layers.append(DenseBlock(32, num_layers=2, growth_rate=16))
    layers.append(nn.Conv1d(32 + 2 * 16, 32, kernel_size=1, stride=1))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

    # Dense Block 2
    layers.append(DenseBlock(32, num_layers=7, growth_rate=16))
    layers.append(nn.Conv1d(32 + 7 * 16, 16, kernel_size=1, stride=1))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool1d(kernel_size=2, stride=4, padding=0))

    # Dense Block 3
    layers.append(DenseBlock(16, num_layers=2, growth_rate=8))
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

    return nn.Sequential(*layers)