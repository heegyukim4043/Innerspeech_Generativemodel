"""
Evaluation: EEG → Mel-Spectrogram
Metrics: MSE, L1, Spectral Convergence, MCD (Mel Cepstral Distortion), SSIM
Visualization: predicted vs ground-truth mel-spectrogram
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader

from data_loader import KaraOneDataset, MEL_CONFIG
from model import EEGToMelModel
from train import CFG


CHECKPOINT = os.path.join(CFG['save_dir'], 'best_model.pt')
RESULT_DIR = "g:/speech_imagery/karaone/results"


# ─── Metrics ─────────────────────────────────────────────────────────────────

def mse(pred, target):
    return F.mse_loss(pred, target).item()

def l1(pred, target):
    return F.l1_loss(pred, target).item()

def spectral_convergence(pred, target):
    return (torch.norm(target - pred, p='fro') /
            (torch.norm(target, p='fro') + 1e-8)).item()

def mel_cepstral_distortion(pred_np, target_np):
    """
    MCD between two mel-spectrograms.
    Lower is better; ~0 for identical.
    pred_np, target_np: (n_mels, T)
    """
    # Convert to MFCC-like via DCT approximation
    from scipy.fftpack import dct
    def to_mfcc(mel):
        log_mel = np.log(mel + 1e-8)
        return dct(log_mel, type=2, axis=0, norm='ortho')[:13]  # 13 cepstral coeffs

    pred_mfcc   = to_mfcc(pred_np)    # (13, T)
    target_mfcc = to_mfcc(target_np)  # (13, T)

    T = min(pred_mfcc.shape[1], target_mfcc.shape[1])
    diff = pred_mfcc[:, :T] - target_mfcc[:, :T]
    mcd  = (10.0 / np.log(10)) * np.sqrt(2.0 * np.sum(diff**2, axis=0)).mean()
    return mcd

def cosine_similarity_score(pred, target):
    """Average cosine similarity per frame."""
    pred_np   = pred.squeeze(0).cpu().numpy()    # (n_mels, T)
    target_np = target.squeeze(0).cpu().numpy()
    T = min(pred_np.shape[1], target_np.shape[1])
    sims = [1 - cosine(pred_np[:, t], target_np[:, t]) for t in range(T)]
    return float(np.mean(sims))


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_mel_comparison(pred_np, target_np, label, trial_idx, save_path):
    """Side-by-side mel-spectrogram: ground truth vs predicted."""
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[4, 1])

    sr, hop = MEL_CONFIG['sr'], MEL_CONFIG['hop_length']

    ax0 = fig.add_subplot(gs[0, 0])
    librosa.display.specshow(target_np, sr=sr, hop_length=hop,
                             x_axis='time', y_axis='mel', ax=ax0, cmap='magma')
    ax0.set_title(f'Ground Truth  [{label}]', fontsize=12)
    ax0.set_ylabel('Mel freq')

    ax1 = fig.add_subplot(gs[0, 1])
    img = librosa.display.specshow(pred_np, sr=sr, hop_length=hop,
                                   x_axis='time', y_axis='mel', ax=ax1, cmap='magma')
    ax1.set_title(f'Predicted  [trial {trial_idx}]', fontsize=12)
    ax1.set_ylabel('')
    plt.colorbar(img, ax=[ax0, ax1], format='%+2.0f dB')

    # Difference
    ax2 = fig.add_subplot(gs[1, :])
    T = min(pred_np.shape[1], target_np.shape[1])
    diff = np.abs(pred_np[:, :T] - target_np[:, :T]).mean(axis=0)
    ax2.plot(diff, color='tomato', linewidth=0.8)
    ax2.set_title('Mean absolute difference per frame', fontsize=10)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('MAD')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_metrics_summary(metrics_per_trial, save_path):
    """Bar chart of per-trial metrics."""
    keys  = ['mse', 'l1', 'sc', 'mcd', 'cos_sim']
    means = {k: np.mean([m[k] for m in metrics_per_trial]) for k in keys}
    stds  = {k: np.std ([m[k] for m in metrics_per_trial]) for k in keys}

    fig, axes = plt.subplots(1, len(keys), figsize=(16, 4))
    for ax, k in zip(axes, keys):
        ax.bar([k], [means[k]], yerr=[stds[k]], color='steelblue', capsize=5)
        ax.set_title(k.upper())
        ax.set_ylabel('value')
    plt.suptitle('Evaluation Metrics (mean ± std across trials)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    return means, stds


# ─── Main Evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = KaraOneDataset(
        eeg_duration_sec = CFG['eeg_duration'],
        mel_time_frames  = CFG['mel_time_frames'],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load model
    print("Loading model checkpoint...")
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    n_channels = dataset[0][0].shape[0]
    model = EEGToMelModel(
        n_channels   = n_channels,
        d_model      = CFG['d_model'],
        n_heads      = CFG['n_heads'],
        n_enc_layers = CFG['n_enc_layers'],
        ff_dim       = CFG['ff_dim'],
        patch_size   = CFG['patch_size'],
        n_mels       = CFG['n_mels'],
        d_state      = CFG['d_state'],
        dropout      = 0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    labels_list = dataset.labels

    metrics_all = []
    print("\nEvaluating trials...")
    for idx, (eeg, mel_gt, label_idx) in enumerate(loader):
        eeg    = eeg.to(device)
        mel_gt = mel_gt.to(device)

        mel_pred = model(eeg)

        # Align time
        T_pred, T_gt = mel_pred.shape[-1], mel_gt.shape[-1]
        if T_pred != T_gt:
            mel_pred = F.interpolate(mel_pred, size=T_gt, mode='linear', align_corners=False)

        pred_np   = mel_pred[0].cpu().numpy()   # (n_mels, T)
        target_np = mel_gt[0].cpu().numpy()

        m = {
            'mse'    : mse(mel_pred, mel_gt),
            'l1'     : l1(mel_pred, mel_gt),
            'sc'     : spectral_convergence(mel_pred, mel_gt),
            'mcd'    : mel_cepstral_distortion(pred_np, target_np),
            'cos_sim': cosine_similarity_score(mel_pred, mel_gt),
            'label'  : labels_list[idx],
        }
        metrics_all.append(m)

        # Save comparison plot for every 10th trial
        if idx % 10 == 0:
            plot_mel_comparison(
                pred_np, target_np,
                label=labels_list[idx], trial_idx=idx,
                save_path=os.path.join(RESULT_DIR, f'mel_trial_{idx:03d}.png')
            )

    # Summary
    means, stds = plot_metrics_summary(
        metrics_all,
        os.path.join(RESULT_DIR, 'metrics_summary.png')
    )

    print("\n═══ Evaluation Summary ═══")
    for k in ['mse', 'l1', 'sc', 'mcd', 'cos_sim']:
        print(f"  {k.upper():10s}: {means[k]:.4f} ± {stds[k]:.4f}")

    # Per-class breakdown
    print("\n═══ Per-Class Results ═══")
    all_labels = sorted(set(m['label'] for m in metrics_all))
    for lbl in all_labels:
        sub = [m for m in metrics_all if m['label'] == lbl]
        avg_mcd = np.mean([m['mcd'] for m in sub])
        avg_cos = np.mean([m['cos_sim'] for m in sub])
        print(f"  {lbl:12s}: MCD={avg_mcd:.3f}  CosSim={avg_cos:.3f}  (n={len(sub)})")

    np.save(os.path.join(RESULT_DIR, 'metrics_all.npy'), metrics_all)
    print(f"\nResults saved to: {RESULT_DIR}")


if __name__ == "__main__":
    evaluate()
