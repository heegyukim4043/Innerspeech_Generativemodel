"""
Training: EEG → Mel-Spectrogram (Transformer + S4-UNet)
Loss: Multi-resolution Spectral + L1 + Spectral Convergence
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data_loader import KaraOneDataset
from model import EEGToMelModel
from model_mamba import EEGToMelMamba


# ─── Config ──────────────────────────────────────────────────────────────────
CFG = dict(
    # Data
    eeg_duration   = 4.0,
    mel_time_frames= 128,
    n_mels         = 80,
    use_speaking   = True,      # thinking + speaking EEG → ~2× 데이터
    augment        = True,      # EEG aug.
    # Model
    d_model        = 96,        # 64 → 96 (330 샘플 대응)
    n_heads        = 4,
    n_enc_layers   = 3,
    ff_dim         = 192,
    patch_size     = 10,
    d_state        = 16,
    dropout        = 0.25,
    # Training
    epochs         = 200,
    batch_size     = 16,
    lr             = 3e-4,
    weight_decay   = 1e-3,
    val_split      = 0.2,
    seed           = 2026,
    # Loss weights
    w_l1           = 1.0,
    w_sc           = 0.5,
    w_mr           = 1.0,       # multi-resolution spectral loss
    # Architecture: "transformer_s4" or "mamba"
    arch           = "mamba",
    # Paths
    save_dir       = "g:/speech_imagery/karaone/checkpoints",
    log_interval   = 10,
)


# ─── Loss Functions ──────────────────────────────────────────────────────────

def spectral_convergence_loss(pred, target):
    return torch.norm(target - pred, p='fro') / (torch.norm(target, p='fro') + 1e-8)


def multi_resolution_spectral_loss(pred, target):
    """
    Multi-resolution spectral loss: L1 + log L1 at multiple time scales.
    pred, target: (B, n_mels, T)
    """
    loss = 0.0
    # 3 temporal resolutions via average pooling
    for scale in [1, 2, 4]:
        if scale > 1:
            p = F.avg_pool1d(pred,   kernel_size=scale, stride=scale)
            t = F.avg_pool1d(target, kernel_size=scale, stride=scale)
        else:
            p, t = pred, target
        # L1 on linear scale
        loss += F.l1_loss(p, t)
        # L1 on log scale
        loss += F.l1_loss(torch.log(p.clamp(min=1e-5)),
                          torch.log(t.clamp(min=1e-5)))
    return loss / 3.0


def combined_loss(pred, target, cfg):
    T_pred, T_gt = pred.shape[-1], target.shape[-1]
    if T_pred != T_gt:
        pred = F.interpolate(pred, size=T_gt, mode='linear', align_corners=False)

    l1 = F.l1_loss(pred, target)
    sc = spectral_convergence_loss(pred, target)
    mr = multi_resolution_spectral_loss(pred, target)

    total = cfg['w_l1'] * l1 + cfg['w_sc'] * sc + cfg['w_mr'] * mr
    return total, {'l1': l1.item(), 'sc': sc.item(), 'mr': mr.item()}


# ─── Training Loop ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, cfg):
    model.train()
    total_loss = 0.0
    for eeg, mel_gt, _ in loader:
        eeg, mel_gt = eeg.to(device), mel_gt.to(device)
        optimizer.zero_grad()
        mel_pred = model(eeg)
        loss, _ = combined_loss(mel_pred, mel_gt, cfg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    sub_losses = {'l1': 0.0, 'sc': 0.0, 'mr': 0.0}
    for eeg, mel_gt, _ in loader:
        eeg, mel_gt = eeg.to(device), mel_gt.to(device)
        mel_pred = model(eeg)
        loss, subs = combined_loss(mel_pred, mel_gt, cfg)
        total_loss += loss.item()
        for k in sub_losses:
            sub_losses[k] += subs[k]
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in sub_losses.items()}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(CFG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(CFG['save_dir'], exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset...")
    full_ds = KaraOneDataset(
        eeg_duration_sec = CFG['eeg_duration'],
        mel_time_frames  = CFG['mel_time_frames'],
        use_speaking     = CFG['use_speaking'],
        augment          = False,   # augment은 train split에만
    )

    n_val   = max(1, int(len(full_ds) * CFG['val_split']))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(CFG['seed'])
    )

    # train split에만 augmentation 적용
    train_ds.dataset.augment = False   # 기본 끄고
    class AugSubset(torch.utils.data.Dataset):
        def __init__(self, subset):
            self.subset = subset
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            eeg, mel, label = self.subset[idx]
            # augment_eeg 직접 호출
            from data_loader import augment_eeg
            eeg_np = augment_eeg(eeg.numpy()) if CFG['augment'] else eeg.numpy()
            return torch.tensor(eeg_np, dtype=torch.float32), mel, label

    train_aug = AugSubset(train_ds)
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_aug, batch_size=CFG['batch_size'], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,    batch_size=CFG['batch_size'], shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Building model...")
    n_channels = full_ds[0][0].shape[0]
    print(f"EEG channels: {n_channels}, arch: {CFG['arch']}")

    if CFG['arch'] == 'mamba':
        model = EEGToMelMamba(
            n_channels   = n_channels,
            d_model      = CFG['d_model'],
            n_enc_layers = CFG['n_enc_layers'],
            patch_size   = CFG['patch_size'],
            n_mels       = CFG['n_mels'],
            d_state      = CFG['d_state'],
            dropout      = CFG['dropout'],
        ).to(device)
    else:
        model = EEGToMelModel(
            n_channels   = n_channels,
            d_model      = CFG['d_model'],
            n_heads      = CFG['n_heads'],
            n_enc_layers = CFG['n_enc_layers'],
            ff_dim       = CFG['ff_dim'],
            patch_size   = CFG['patch_size'],
            n_mels       = CFG['n_mels'],
            d_state      = CFG['d_state'],
            dropout      = CFG['dropout'],
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    # Warm restarts: 50 epoch 주기로 LR 리셋 → 지역 최솟값 탈출
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-6)

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n[3/4] Training...\n")
    best_val = float('inf')
    history  = {'train': [], 'val': []}
    patience, patience_cnt = 80, 0

    for epoch in range(1, CFG['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, CFG)
        val_loss, subs = val_epoch(model, val_loader, device, CFG)
        scheduler.step()
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if epoch % CFG['log_interval'] == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{CFG['epochs']} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"L1:{subs['l1']:.3f} SC:{subs['sc']:.3f} MR:{subs['mr']:.3f} | "
                  f"LR: {lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': best_val,
                'cfg': CFG,
            }, os.path.join(CFG['save_dir'], f"best_model_{CFG['arch']}.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest val loss: {best_val:.4f}")
    np.save(os.path.join(CFG['save_dir'], 'history.npy'), history)
    print("[4/4] Training complete.")


if __name__ == "__main__":
    main()
