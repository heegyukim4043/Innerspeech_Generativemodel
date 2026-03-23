"""
EEG → Mel-Spectrogram: Mamba (S6) 기반 모델
Architecture: Bidirectional Mamba Encoder + Mamba-UNet Decoder

S4 vs Mamba 차이:
  S4  : A, B, C 고정 → 입력과 무관한 convolution kernel
  Mamba: B, C, Δ 가 입력에 따라 변함 (selective SSM, S6)
         → 중요한 시점/채널에 집중 가능
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ═══════════════════════════════════════════════════════════════════════════
# Mamba Block (S6 — Selective State Space)
# ═══════════════════════════════════════════════════════════════════════════

class MambaBlock(nn.Module):
    """
    Mamba block (Gu & Dao, 2023).
    핵심: B, C, Δ 가 입력 x에 의존 → 선택적으로 정보 통합

    x: (B, L, d_model) → (B, L, d_model)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)  # expanded inner dim

        # ── Input projection (gated) ─────────────────────────────────────
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # ── Depthwise causal conv (local context mixing) ─────────────────
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,           # causal: trim later
            groups=self.d_inner,
            bias=True,
        )

        # ── SSM projections (input-dependent B, C, Δ) ────────────────────
        # Δ: (B,L,d_inner),  B: (B,L,d_state),  C: (B,L,d_state)
        self.x_proj  = nn.Linear(self.d_inner, 1 + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A: log-parameterized, initialized with HiPPO-like spacing
        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))   # (d_inner, d_state)
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        # ── Output ───────────────────────────────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)

        nn.init.uniform_(self.dt_proj.bias, -4.0, -1.0)   # Δ > 0 초기화

    def _selective_scan(self, x, delta, A, B, C):
        """
        Discretized SSM scan.
        x    : (B, L, d_inner)
        delta: (B, L, d_inner)
        A    : (d_inner, d_state)   — continuous A
        B    : (B, L, d_state)      — input-dependent
        C    : (B, L, d_state)      — input-dependent
        returns y: (B, L, d_inner)
        """
        B_sz, L, d = x.shape
        N = self.d_state

        # Discretize: zero-order hold
        # dA: (B, L, d_inner, d_state)
        dA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        # dB: (B, L, d_inner, d_state)
        dB = torch.einsum('bld,bln->bldn', delta, B)

        # Sequential scan  O(L·d·N)
        # For L≤500 this is fast enough; replace with parallel scan for L>1000
        h = torch.zeros(B_sz, d, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            # h: (B, d_inner, d_state)
            h = dA[:, t] * h + dB[:, t] * x[:, t, :].unsqueeze(-1)
            # y_t = C_t · h_t  → (B, d_inner)
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)               # (B, L, d_inner)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        return y

    def forward(self, x):
        """x: (B, L, d_model)"""
        residual = x
        B_sz, L, _ = x.shape

        # 1. Gate projection
        xz      = self.in_proj(x)                          # (B, L, d_inner*2)
        x_inner, z = xz.chunk(2, dim=-1)                  # (B, L, d_inner) each

        # 2. Causal depthwise conv
        x_conv = x_inner.transpose(1, 2)                  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :L]              # causal trim
        x_conv = F.silu(x_conv.transpose(1, 2))           # (B, L, d_inner)

        # 3. Input-dependent SSM params
        x_dbl = self.x_proj(x_conv)                       # (B, L, 1+N*2)
        dt, B_ssm, C_ssm = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                 # (B, L, d_inner), Δ>0

        A = -torch.exp(self.A_log.float())                 # (d_inner, d_state)

        # 4. Selective scan
        y = self._selective_scan(x_conv, dt, A, B_ssm, C_ssm)  # (B, L, d_inner)

        # 5. Gating
        y = y * F.silu(z)

        out = self.drop(self.out_proj(y))
        return self.norm(out + residual)


class BidirectionalMambaBlock(nn.Module):
    """
    EEG 분석을 위한 양방향 Mamba:
    forward scan + backward scan → concat → project
    미래 정보도 활용 가능 (인과성 불필요한 인코더용)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.fwd = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.bwd = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        f = self.fwd(x)                                  # (B, L, d)
        b = self.bwd(torch.flip(x, dims=[1]))
        b = torch.flip(b, dims=[1])                      # (B, L, d)
        return self.norm(self.proj(torch.cat([f, b], dim=-1)))


# ═══════════════════════════════════════════════════════════════════════════
# Mamba Encoder (EEG → latent)
# ═══════════════════════════════════════════════════════════════════════════

class EEGMambaEncoder(nn.Module):
    """
    Spatial patch embedding + Bidirectional Mamba layers.
    EEG (B, C, T) → (B, T', d_model)
    """
    def __init__(self, n_channels=62, d_model=96, n_layers=4,
                 patch_size=10, d_state=16, d_conv=4, dropout=0.1):
        super().__init__()
        # Spatial + temporal embedding via strided conv
        self.patch_embed = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, 2000, d_model) * 0.02)
        self.layers  = nn.ModuleList([
            BidirectionalMambaBlock(d_model, d_state, d_conv, expand=2, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, C, T) → (B, T', d_model)"""
        x = self.patch_embed(x)                          # (B, d_model, T')
        x = rearrange(x, 'b d t -> b t d')
        T = x.shape[1]
        x = x + self.pos_enc[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════════════
# Mamba-UNet Decoder
# ═══════════════════════════════════════════════════════════════════════════

class MambaUNetBlock(nn.Module):
    """Mamba block + optional temporal resampling."""
    def __init__(self, d_in, d_out, d_state=16, scale=1, dropout=0.1):
        super().__init__()
        self.mamba = MambaBlock(d_in, d_state=d_state, dropout=dropout)
        self.proj  = nn.Linear(d_in, d_out)
        self.norm  = nn.LayerNorm(d_out)
        self.scale = scale

    def forward(self, x):
        """x: (B, L, d_in) → (B, L*scale, d_out)"""
        x = self.mamba(x)
        x = self.proj(x)
        if self.scale != 1:
            x = rearrange(x, 'b l d -> b d l')
            x = F.interpolate(x, scale_factor=self.scale, mode='linear', align_corners=False)
            x = rearrange(x, 'b d l -> b l d')
        return self.norm(x)


class MambaUNet(nn.Module):
    """
    Mamba 기반 UNet decoder.
    skip connection으로 멀티스케일 특징 보존.
    """
    def __init__(self, d_model=96, n_mels=80, d_state=16, dropout=0.1):
        super().__init__()
        # Encoder path (downsample)
        self.enc1 = MambaUNetBlock(d_model, 256,  d_state, scale=1,   dropout=dropout)
        self.enc2 = MambaUNetBlock(256,     128,  d_state, scale=0.5, dropout=dropout)
        self.enc3 = MambaUNetBlock(128,     64,   d_state, scale=0.5, dropout=dropout)

        # Bottleneck
        self.bottleneck = MambaBlock(64, d_state=d_state, dropout=dropout)

        # Decoder path (upsample + skip)
        self.dec3 = MambaUNetBlock(64  + 128,     128, d_state, scale=2.0, dropout=dropout)
        self.dec2 = MambaUNetBlock(128 + 256,     256, d_state, scale=2.0, dropout=dropout)
        self.dec1 = MambaUNetBlock(256 + d_model, 256, d_state, scale=1,   dropout=dropout)

        # Output
        self.out_proj = nn.Sequential(
            nn.Linear(256, n_mels),
            nn.Sigmoid(),
        )

    @staticmethod
    def _align(a, ref):
        """시간 축 길이를 ref에 맞춤."""
        Ta, Tr = a.shape[1], ref.shape[1]
        if Ta > Tr:
            return a[:, :Tr, :]
        elif Ta < Tr:
            pad = torch.zeros(a.shape[0], Tr - Ta, a.shape[2], device=a.device)
            return torch.cat([a, pad], dim=1)
        return a

    def forward(self, x):
        """x: (B, T', d_model) → (B, n_mels, T_out)"""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bottleneck(e3)

        d3 = self.dec3(torch.cat([self._align(b,  e2), e2], dim=-1))
        d2 = self.dec2(torch.cat([self._align(d3, e1), e1], dim=-1))
        d1 = self.dec1(torch.cat([self._align(d2, x),  x ], dim=-1))

        out = self.out_proj(d1)                           # (B, T', n_mels)
        return rearrange(out, 'b t m -> b m t')           # (B, n_mels, T')


# ═══════════════════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════════════════

class EEGToMelMamba(nn.Module):
    """
    Full Mamba pipeline:
      EEG (B, C, T) → BiMamba Encoder → Mamba-UNet → Mel (B, 80, T')

    vs 기존 Transformer+S4:
      - Transformer attention  →  Bidirectional Mamba  (선택적 집중)
      - S4 UNet               →  Mamba UNet            (입력 의존 상태 전이)
    """
    def __init__(
        self,
        n_channels  = 62,
        d_model     = 96,
        n_enc_layers= 4,
        patch_size  = 10,
        n_mels      = 80,
        d_state     = 16,
        d_conv      = 4,
        dropout     = 0.1,
    ):
        super().__init__()
        self.encoder = EEGMambaEncoder(
            n_channels, d_model, n_enc_layers, patch_size, d_state, d_conv, dropout
        )
        self.decoder = MambaUNet(d_model, n_mels, d_state, dropout)

    def forward(self, eeg):
        feat     = self.encoder(eeg)      # (B, T', d_model)
        mel_pred = self.decoder(feat)     # (B, n_mels, T_pred)
        return mel_pred


if __name__ == "__main__":
    model = EEGToMelMamba()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Mamba model parameters: {n_params:,}")

    eeg = torch.randn(2, 62, 4000)
    mel = model(eeg)
    print(f"Input  EEG : {eeg.shape}")
    print(f"Output Mel : {mel.shape}")

    # 기존 Transformer+S4 모델과 비교
    from model import EEGToMelModel
    model_s4 = EEGToMelModel(n_channels=62, d_model=96)
    n_s4 = sum(p.numel() for p in model_s4.parameters() if p.requires_grad)
    print(f"\n비교:")
    print(f"  Transformer+S4  : {n_s4:,} params")
    print(f"  Mamba (S6)      : {n_params:,} params")
