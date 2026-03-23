"""
KaraOne Dataset Loader
EEG (thinking + speaking imagined speech) + wav (vocalized speech → mel-spectrogram)
"""
import os
import numpy as np
import scipy.io as sio
import scipy.io.wavfile as wavfile
import mne
import librosa
import torch
from torch.utils.data import Dataset

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_ROOT = "g:/speech_imagery/karaone/p/spoclab/users/szhao/EEG/data/MM05"
EEG_FILE  = os.path.join(DATA_ROOT, "Acquisition 232 Data.set")   # ICA 전처리 완료본
EPOCH_MAT = os.path.join(DATA_ROOT, "epoch_inds.mat")
KINECT_DIR= os.path.join(DATA_ROOT, "kinect_data")
LABEL_TXT = os.path.join(KINECT_DIR, "labels.txt")

# ─── Mel-spectrogram config ─────────────────────────────────────────────────
MEL_CONFIG = dict(
    sr        = 16000,
    n_fft     = 512,
    hop_length= 128,
    n_mels    = 80,
    fmin      = 50,
    fmax      = 8000,
)

# ─── EEG config ─────────────────────────────────────────────────────────────
EEG_SFREQ   = 1000   # Hz


def load_labels():
    with open(LABEL_TXT) as f:
        labels = [l.strip() for l in f.readlines() if l.strip()]
    unique = sorted(set(labels))
    label2idx = {l: i for i, l in enumerate(unique)}
    return labels, label2idx


def wav_to_mel(wav_path, config=MEL_CONFIG):
    """Load wav and convert to log mel-spectrogram."""
    sr, data = wavfile.read(wav_path)
    audio = data.astype(np.float32) / 32768.0

    if sr != config['sr']:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config['sr'])

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config['sr'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels'],
        fmin=config['fmin'],
        fmax=config['fmax'],
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)   # (n_mels, T_mel)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel.astype(np.float32)


def load_eeg_raw():
    """Load preprocessed EEG with MNE."""
    raw = mne.io.read_raw_eeglab(EEG_FILE, preload=True, verbose=False)
    raw.filter(1.0, 40.0, verbose=False)
    raw.pick_types(eeg=True, verbose=False)
    return raw


def load_epoch_indices():
    mat = sio.loadmat(EPOCH_MAT)
    thinking = np.array([m.flatten() for m in mat['thinking_inds'].flatten()])  # (165, 2)
    speaking  = np.array([m.flatten() for m in mat['speaking_inds'].flatten()])  # (330, 2)
    speaking  = speaking[1::2]   # (165, 2) — 실제 발화 구간만
    return thinking, speaking


# ─── EEG Augmentation ────────────────────────────────────────────────────────

def augment_eeg(eeg, sfreq=1000):
    """
    eeg: (C, T) numpy array
    Returns augmented (C, T) numpy array
    """
    C, T = eeg.shape

    # 1. Gaussian noise (SNR ~20dB)
    if np.random.rand() < 0.5:
        noise = np.random.randn(C, T) * 0.05 * eeg.std()
        eeg = eeg + noise

    # 2. Time shift (±100ms)
    if np.random.rand() < 0.5:
        shift = np.random.randint(-100, 100)
        eeg = np.roll(eeg, shift, axis=1)

    # 3. Channel dropout (5% of channels zeroed)
    if np.random.rand() < 0.5:
        n_drop = max(1, int(C * 0.05))
        drop_idx = np.random.choice(C, n_drop, replace=False)
        eeg = eeg.copy()
        eeg[drop_idx, :] = 0.0

    # 4. Amplitude scaling (±20%)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.8, 1.2)
        eeg = eeg * scale

    return eeg


class KaraOneDataset(Dataset):
    """
    Returns pairs of:
      eeg   : (C, T_eeg)      — EEG segment (imagined or pronounced)
      mel_gt: (n_mels, T_mel) — vocalized speech mel-spectrogram (ground truth)
      label : int

    use_speaking=True 이면 thinking + speaking EEG 모두 사용 (~2× 데이터)
    augment=True 이면 EEG 증강 적용
    """
    def __init__(self, eeg_duration_sec=4.0, mel_time_frames=None,
                 use_speaking=True, augment=False):
        super().__init__()
        self.augment       = augment
        self.mel_frames    = mel_time_frames

        print("Loading EEG raw data...")
        self.raw   = load_eeg_raw()
        self.sfreq = int(self.raw.info['sfreq'])
        self.eeg_len = int(eeg_duration_sec * self.sfreq)

        print("Loading epoch indices...")
        thinking_inds, speaking_inds = load_epoch_indices()

        print("Loading labels...")
        self.labels, self.label2idx = load_labels()

        # Build trial list: (eeg_start, eeg_end, wav_idx)
        # wav idx = trial index in kinect_data/N.wav
        self.trials = []
        for i in range(len(thinking_inds)):
            self.trials.append((int(thinking_inds[i,0]), int(thinking_inds[i,1]), i))

        if use_speaking:
            for i in range(len(speaking_inds)):
                self.trials.append((int(speaking_inds[i,0]), int(speaking_inds[i,1]), i))

        print(f"Dataset ready: {len(self.trials)} trials "
              f"({'thinking+speaking' if use_speaking else 'thinking only'}), "
              f"EEG {self.eeg_len} samples ({eeg_duration_sec}s), "
              f"augment={augment}")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        start, end, wav_idx = self.trials[idx]

        # ── EEG ──────────────────────────────────────────────────────────────
        eeg_data = self.raw.get_data(start=start, stop=end)  # (C, T)

        T = eeg_data.shape[1]
        if T >= self.eeg_len:
            eeg_data = eeg_data[:, :self.eeg_len]
        else:
            pad = np.zeros((eeg_data.shape[0], self.eeg_len - T), dtype=np.float32)
            eeg_data = np.concatenate([eeg_data, pad], axis=1)

        # z-score per channel
        mu  = eeg_data.mean(axis=1, keepdims=True)
        std = eeg_data.std(axis=1,  keepdims=True) + 1e-8
        eeg_data = (eeg_data - mu) / std

        # Augmentation (train only)
        if self.augment:
            eeg_data = augment_eeg(eeg_data, self.sfreq)

        # ── Mel GT ───────────────────────────────────────────────────────────
        wav_path = os.path.join(KINECT_DIR, f"{wav_idx}.wav")
        mel_gt = wav_to_mel(wav_path)

        if self.mel_frames is not None:
            T_mel = mel_gt.shape[1]
            if T_mel >= self.mel_frames:
                mel_gt = mel_gt[:, :self.mel_frames]
            else:
                pad = np.zeros((mel_gt.shape[0], self.mel_frames - T_mel), dtype=np.float32)
                mel_gt = np.concatenate([mel_gt, pad], axis=1)

        # ── Label ─────────────────────────────────────────────────────────────
        label = self.label2idx[self.labels[wav_idx]]

        return (
            torch.tensor(eeg_data, dtype=torch.float32),
            torch.tensor(mel_gt,   dtype=torch.float32),
            torch.tensor(label,    dtype=torch.long),
        )


if __name__ == "__main__":
    ds = KaraOneDataset(eeg_duration_sec=4.0, mel_time_frames=128,
                        use_speaking=True, augment=True)
    eeg, mel, label = ds[0]
    print(f"EEG shape : {eeg.shape}")
    print(f"Mel shape : {mel.shape}")
    print(f"Total trials: {len(ds)}")
