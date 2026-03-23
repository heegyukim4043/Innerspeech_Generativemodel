"""
EEG → Mel-Spectrogram Generation Model
Architecture: Transformer Encoder + S4-UNet Decoder
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ═══════════════════════════════════════════════════════════════════════════
# S4 (Simplified Diagonal State Space Sequence Model)
# ═══════════════════════════════════════════════════════════════════════════

class S4Layer(nn.Module):
    """
    Simplified S4 using diagonal complex SSM.
    State: h_t = A h_{t-1} + B x_t
    Output: y_t = Re(C h_t) + D x_t
    A parameterized as diagonal complex (HiPPO-inspired init).
    """
    def __init__(self, d_model, d_state=64, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal A in log space: A = -exp(log_A) + i * imag_A (stable)
        self.log_A_real = nn.Parameter(torch.randn(d_model, d_state))
        self.A_imag     = nn.Parameter(torch.randn(d_model, d_state))

        # B, C projections
        self.B = nn.Parameter(torch.randn(d_model, d_state, 2))  # complex
        self.C = nn.Parameter(torch.randn(d_model, d_state, 2))  # complex

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(d_model))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(d_model)

        nn.init.normal_(self.B, std=0.01)
        nn.init.normal_(self.C, std=0.01)

    def _build_kernel(self, L):
        """Compute convolution kernel of length L from SSM parameters."""
        # A: (d_model, d_state) complex
        A_real = -torch.exp(self.log_A_real)                # negative for stability
        A = torch.complex(A_real, self.A_imag)              # (d_model, d_state)

        B = torch.view_as_complex(self.B.contiguous())      # (d_model, d_state)
        C = torch.view_as_complex(self.C.contiguous())      # (d_model, d_state)

        # Powers of A: A^0, A^1, ..., A^{L-1}
        # shape: (L, d_model, d_state)
        powers = torch.arange(L, device=A.device).unsqueeze(-1).unsqueeze(-1).float()
        A_pow  = torch.exp(powers * A.unsqueeze(0))         # (L, d_model, d_state)

        # Kernel k_t = Re(C * A^t * B)  → (d_model, L)
        K = torch.einsum('ds,lds,ds->dl', C.conj(), A_pow, B).real  # (d_model, L)
        return K

    def forward(self, x):
        """
        x: (B, L, d_model)
        returns: (B, L, d_model)
        """
        residual = x
        B, L, d = x.shape

        # Build SSM kernel
        K = self._build_kernel(L)       # (d_model, L)

        # Causal convolution via FFT
        x_t  = rearrange(x, 'b l d -> b d l')          # (B, d, L)
        K_fl = torch.flip(K, dims=[-1])                  # causal flip

        fft_size = 2 * L
        X_f = torch.fft.rfft(x_t,   n=fft_size)
        K_f = torch.fft.rfft(K_fl,  n=fft_size)
        Y_f = X_f * K_f.unsqueeze(0)
        y   = torch.fft.irfft(Y_f, n=fft_size)[..., L-1:2*L-1]  # (B, d, L)

        # Skip connection D
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * x_t
        y = rearrange(y, 'b d l -> b l d')

        y = self.dropout(self.out_proj(y))
        return self.norm(y + residual)


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Encoder (EEG feature extractor)
# ═══════════════════════════════════════════════════════════════════════════

class EEGSpatialEmbed(nn.Module):
    """Project 64 EEG channels → d_model with spatial mixing."""
    def __init__(self, n_channels=64, d_model=256, patch_size=10):
        super().__init__()
        self.patch_size = patch_size
        # Temporal patch embedding: Conv1D with stride = patch_size
        self.proj = nn.Conv1d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, C, T) → (B, T//patch, d_model)"""
        x = self.proj(x)                    # (B, d_model, T//patch)
        x = rearrange(x, 'b d t -> b t d')
        return self.norm(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        # Feed-forward
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class EEGTransformerEncoder(nn.Module):
    def __init__(self, n_channels=64, d_model=256, n_heads=8,
                 n_layers=4, ff_dim=512, patch_size=10, dropout=0.1):
        super().__init__()
        self.embed   = EEGSpatialEmbed(n_channels, d_model, patch_size)
        self.pos_enc = nn.Parameter(torch.randn(1, 2000, d_model) * 0.02)
        self.layers  = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, 64, T_eeg) → (B, T', d_model)"""
        x = self.embed(x)                               # (B, T', d_model)
        T = x.shape[1]
        x = x + self.pos_enc[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════════════
# S4-UNet Decoder (generates mel-spectrogram)
# ═══════════════════════════════════════════════════════════════════════════

class S4UNetBlock(nn.Module):
    """S4 + optional upsampling/downsampling."""
    def __init__(self, d_in, d_out, d_state=32, scale=1, dropout=0.1):
        super().__init__()
        self.s4   = S4Layer(d_in, d_state=d_state, dropout=dropout)
        self.proj = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.scale = scale   # >1: upsample, <1: downsample (as fraction)

    def forward(self, x):
        """x: (B, L, d_in) → (B, L*scale, d_out)"""
        x = self.s4(x)
        x = self.proj(x)
        if self.scale != 1:
            # Interpolate along time dimension
            x = rearrange(x, 'b l d -> b d l')
            x = F.interpolate(x, scale_factor=self.scale, mode='linear', align_corners=False)
            x = rearrange(x, 'b d l -> b l d')
        return self.norm(x)


class S4UNet(nn.Module):
    """
    UNet with S4 layers.
    Encoder: downsample × 3
    Bottleneck: S4
    Decoder: upsample × 3 with skip connections
    Output: mel-spectrogram (B, n_mels, T_mel)
    """
    def __init__(self, d_model=256, n_mels=80, d_state=32, dropout=0.1):
        super().__init__()

        # Encoder (downsample)
        self.enc1 = S4UNetBlock(d_model, 256,  d_state, scale=1,   dropout=dropout)
        self.enc2 = S4UNetBlock(256,     128,  d_state, scale=0.5, dropout=dropout)
        self.enc3 = S4UNetBlock(128,     64,   d_state, scale=0.5, dropout=dropout)

        # Bottleneck
        self.bottleneck = S4Layer(64, d_state=d_state, dropout=dropout)

        # Decoder (upsample + skip)
        # dec3 input: bottleneck(64) + enc2 skip(128)
        # dec2 input: dec3 out(128)  + enc1 skip(256)
        # dec1 input: dec2 out(256)  + x skip(d_model)
        self.dec3 = S4UNetBlock(64 + 128,     128, d_state, scale=2.0, dropout=dropout)
        self.dec2 = S4UNetBlock(128 + 256,    256, d_state, scale=2.0, dropout=dropout)
        self.dec1 = S4UNetBlock(256 + d_model, 256, d_state, scale=1,  dropout=dropout)

        # Output head
        self.out_proj = nn.Sequential(
            nn.Linear(256, n_mels),
            nn.Sigmoid(),   # mel values in [0, 1]
        )

    def forward(self, x):
        """
        x: (B, T', d_model)
        returns: (B, n_mels, T_out)
        """
        # Encoder
        e1 = self.enc1(x)   # (B, T',    256)
        e2 = self.enc2(e1)  # (B, T'//2, 128)
        e3 = self.enc3(e2)  # (B, T'//4, 64)

        # Bottleneck
        b = self.bottleneck(e3)   # (B, T'//4, 64)

        # Decoder with skip connections (align time dims)
        def align(a, ref):
            """Trim/pad a to match ref's time dimension."""
            Ta, Tr = a.shape[1], ref.shape[1]
            if Ta > Tr:
                return a[:, :Tr, :]
            elif Ta < Tr:
                pad = torch.zeros(a.shape[0], Tr - Ta, a.shape[2], device=a.device)
                return torch.cat([a, pad], dim=1)
            return a

        d3 = self.dec3(torch.cat([align(b,  e2), e2], dim=-1))
        d2 = self.dec2(torch.cat([align(d3, e1), e1], dim=-1))
        d1 = self.dec1(torch.cat([align(d2, x),  x ], dim=-1))

        out = self.out_proj(d1)              # (B, T', n_mels)
        out = rearrange(out, 'b t m -> b m t')  # (B, n_mels, T')
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════════════════

class EEGToMelModel(nn.Module):
    """
    Full pipeline:
      EEG (B, 64, T_eeg) → Transformer Encoder → S4-UNet → Mel (B, 80, T_mel)
    """
    def __init__(
        self,
        n_channels  = 64,
        d_model     = 256,
        n_heads     = 8,
        n_enc_layers= 4,
        ff_dim      = 512,
        patch_size  = 10,
        n_mels      = 80,
        d_state     = 32,
        dropout     = 0.1,
    ):
        super().__init__()
        self.encoder = EEGTransformerEncoder(
            n_channels, d_model, n_heads, n_enc_layers, ff_dim, patch_size, dropout
        )
        self.decoder = S4UNet(d_model, n_mels, d_state, dropout)

    def forward(self, eeg):
        """
        eeg: (B, 64, T_eeg)
        returns: mel_pred (B, n_mels, T_pred)
        """
        feat     = self.encoder(eeg)      # (B, T', d_model)
        mel_pred = self.decoder(feat)     # (B, n_mels, T_pred)
        return mel_pred


if __name__ == "__main__":
    model = EEGToMelModel()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    eeg = torch.randn(2, 64, 4000)
    mel = model(eeg)
    print(f"Input EEG : {eeg.shape}")
    print(f"Output Mel: {mel.shape}")
