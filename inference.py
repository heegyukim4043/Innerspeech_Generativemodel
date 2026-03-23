"""
Inference: EEG → Mel-Spectrogram → Speech (.wav)
Vocoder: HiFi-GAN (speechbrain pretrained) or Griffin-Lim fallback
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_loader import KaraOneDataset, MEL_CONFIG
from model import EEGToMelModel
from train import CFG

CHECKPOINT  = os.path.join(CFG['save_dir'], 'best_model.pt')
OUTPUT_DIR  = "g:/speech_imagery/karaone/outputs"
VOCODER     = "hifigan"   # "hifigan" or "griffinlim"


# ─── Vocoder ─────────────────────────────────────────────────────────────────

def load_hifigan(device):
    try:
        from speechbrain.inference.vocoders import HIFIGAN
        hifi = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech",
            savedir=os.path.join(OUTPUT_DIR, "pretrained_hifigan"),
            run_opts={"device": str(device)},
        )
        print("Vocoder: HiFi-GAN loaded")
        return hifi
    except Exception as e:
        print(f"HiFi-GAN load failed ({e}), falling back to Griffin-Lim")
        return None


def mel_to_wav_hifigan(mel, hifi_gan, target_sr=16000):
    """
    mel: (n_mels, T) numpy array, normalized [0,1] log-mel
    returns: waveform numpy (T_audio,)
    """
    # HiFi-GAN expects (batch, n_mels, T) float tensor
    mel_t = torch.tensor(mel).unsqueeze(0).float()  # (1, 80, T)

    # HiFi-GAN is trained on 22050Hz — resample output to 16000
    with torch.no_grad():
        wav = hifi_gan.decode_batch(mel_t)           # (1, 1, T_audio) at 22050Hz
    wav_np = wav.squeeze().cpu().numpy()

    # Resample to target_sr
    wav_np = librosa.resample(wav_np, orig_sr=22050, target_sr=target_sr)
    return wav_np


def mel_to_wav_griffinlim(mel, sr=16000, n_fft=512, hop_length=128, n_iter=64):
    """
    mel: (n_mels, T) numpy array, normalized [0,1] log-mel
    returns: waveform numpy (T_audio,)
    """
    # Denormalize: [0,1] → dB → power
    mel_db    = mel * 80.0 - 80.0                         # approx dB scale
    mel_power = librosa.db_to_power(mel_db)               # power mel-spec

    # Mel → linear spectrogram (pseudo-inverse mel filterbank)
    mel_filter = librosa.filters.mel(
        sr=sr, n_fft=n_fft,
        n_mels=mel.shape[0],
        fmin=MEL_CONFIG['fmin'],
        fmax=MEL_CONFIG['fmax'],
    )
    mel_pinv   = np.linalg.pinv(mel_filter)               # (n_fft//2+1, n_mels)
    spec_power = np.maximum(mel_pinv @ mel_power, 0.0)    # (n_fft//2+1, T)

    # Griffin-Lim
    wav = librosa.griffinlim(
        np.sqrt(spec_power),
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=n_fft,
    )
    return wav


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_comparison(pred_mel, gt_mel, label, trial_idx, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sr, hop = MEL_CONFIG['sr'], MEL_CONFIG['hop_length']
    import librosa.display

    librosa.display.specshow(gt_mel,   sr=sr, hop_length=hop,
                             x_axis='time', y_axis='mel',
                             ax=axes[0], cmap='magma')
    axes[0].set_title(f'Ground Truth  [{label}]')

    librosa.display.specshow(pred_mel, sr=sr, hop_length=hop,
                             x_axis='time', y_axis='mel',
                             ax=axes[1], cmap='magma')
    axes[1].set_title(f'Predicted  [trial {trial_idx}]')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ─── Main Inference ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(n_samples=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "wav_pred"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "wav_gt"),   exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "mel_plots"), exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = KaraOneDataset(
        eeg_duration_sec = CFG['eeg_duration'],
        mel_time_frames  = CFG['mel_time_frames'],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model...")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    n_ch = dataset[0][0].shape[0]
    model = EEGToMelModel(
        n_channels   = n_ch,
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
    print(f"Loaded epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # ── Vocoder ───────────────────────────────────────────────────────────────
    hifi_gan = load_hifigan(device) if VOCODER == "hifigan" else None

    # ── Inference loop ────────────────────────────────────────────────────────
    print(f"\nGenerating speech for {n_samples or len(dataset)} trials...\n")
    sr = MEL_CONFIG['sr']

    for idx, (eeg, mel_gt, label_idx) in enumerate(loader):
        if n_samples and idx >= n_samples:
            break

        label = dataset.labels[idx]
        eeg   = eeg.to(device)

        # 1. EEG → mel-spectrogram
        mel_pred = model(eeg)                           # (1, 80, T_pred)

        # Align to GT length
        T_gt = mel_gt.shape[-1]
        mel_pred = F.interpolate(mel_pred, size=T_gt, mode='linear', align_corners=False)

        pred_np = mel_pred[0].cpu().numpy()             # (80, T)
        gt_np   = mel_gt[0].cpu().numpy()               # (80, T)

        # 2. mel → waveform
        if hifi_gan is not None:
            wav_pred = mel_to_wav_hifigan(pred_np, hifi_gan, target_sr=sr)
            wav_gt   = mel_to_wav_hifigan(gt_np,   hifi_gan, target_sr=sr)
        else:
            wav_pred = mel_to_wav_griffinlim(pred_np, sr=sr)
            wav_gt   = mel_to_wav_griffinlim(gt_np,   sr=sr)

        # 3. Save wav files
        label_str = label.replace('/', '').replace(' ', '_')
        wav_pred_t = torch.tensor(wav_pred).unsqueeze(0).float()
        wav_gt_t   = torch.tensor(wav_gt).unsqueeze(0).float()

        torchaudio.save(
            os.path.join(OUTPUT_DIR, "wav_pred", f"{idx:03d}_{label_str}_pred.wav"),
            wav_pred_t, sr
        )
        torchaudio.save(
            os.path.join(OUTPUT_DIR, "wav_gt", f"{idx:03d}_{label_str}_gt.wav"),
            wav_gt_t, sr
        )

        # 4. Save mel comparison plot
        plot_comparison(
            pred_np, gt_np, label, idx,
            save_path=os.path.join(OUTPUT_DIR, "mel_plots", f"{idx:03d}_{label_str}.png")
        )

        print(f"[{idx+1:3d}] {label:12s} | "
              f"pred wav: {len(wav_pred)/sr:.2f}s | "
              f"gt wav: {len(wav_gt)/sr:.2f}s")

    print(f"\nDone. Outputs saved to: {OUTPUT_DIR}")
    print(f"  wav_pred/   ← 모델 생성 음성")
    print(f"  wav_gt/     ← ground truth 음성 (mel→wav 변환)")
    print(f"  mel_plots/  ← mel-spectrogram 비교 이미지")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None, help='Number of trials to generate (default: all)')
    parser.add_argument('--vocoder', type=str, default='hifigan', choices=['hifigan', 'griffinlim'])
    args = parser.parse_args()
    VOCODER = args.vocoder
    run_inference(n_samples=args.n)
