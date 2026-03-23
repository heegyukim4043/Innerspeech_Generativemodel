# EEG-to-Speech: Inner Speech Mel-Spectrogram Generation

Generating mel-spectrograms (and audio) from EEG signals recorded during **imagined/inner speech**, using the [KaraOne dataset](http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html).

Two model architectures are provided:
- **Transformer + S4-UNet** — Multi-head attention encoder + structured state-space UNet decoder
- **Mamba (S6) + Mamba-UNet** — Bidirectional selective SSM encoder + Mamba-UNet decoder

---

## Dataset

### KaraOne
- 14 subjects, 11 classes (7 phonemes + 4 words)
- EEG: 64-channel, 1 kHz (Neuroscan)
- Audio: vocalized speech recorded via Microsoft Kinect (16 kHz `.wav`)
- Imagined + Vocalized speech pairs (~165 trials/subject)

### Download

```bash
# Subject 1 (MM05, ~2.1 GB)
curl -L http://www.cs.toronto.edu/~complingweb/data/karaOne/MM05.tar.bz2 -o MM05.tar.bz2
tar -xjf MM05.tar.bz2

# Other subjects: MM08, MM09, MM10, MM11, MM12, MM14, MM15, MM16, MM18, MM19, MM20, MM21, P02
```

After extraction, update `DATA_ROOT` in [data_loader.py](data_loader.py):
```python
DATA_ROOT = "/path/to/MM05"
```

### Expected Directory Structure

```
MM05/
├── Acquisition 232 Data.set   ← ICA-preprocessed EEG (EEGLAB)
├── epoch_inds.mat             ← Trial indices (thinking / speaking)
└── kinect_data/
    ├── 0.wav ~ 164.wav        ← Vocalized speech (Ground Truth)
    └── labels.txt             ← Per-trial class labels
```

---

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU (recommended) + CUDA 11.8 driver

### Step 1: Create Virtual Environment

**Linux / Mac**
```bash
python -m venv eeg_mel_env
source eeg_mel_env/bin/activate
```

**Windows (PowerShell)**

> If `activate` fails due to execution policy, use the python executable directly (Step 3).

```powershell
python -m venv eeg_mel_env
# Option A: activate normally
eeg_mel_env\Scripts\activate
# Option B: if activation is blocked, skip and use full path instead (see Step 3)
```

### Step 2: Install PyTorch (GPU)

```bash
# CUDA 11.8 (GTX 10xx / RTX 20xx / 30xx / 40xx)
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only (no GPU)
pip install torch==2.4.1 torchvision torchaudio
```

Verify GPU is available:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Expected: CUDA: True / GPU: NVIDIA GeForce ...
```

### Step 3: Install Remaining Packages

```bash
pip install -r requirements.txt
```

**Windows — if venv activation is blocked:**
```powershell
# Use full path to pip/python instead of activating
eeg_mel_env\Scripts\pip.exe install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
eeg_mel_env\Scripts\pip.exe install -r requirements.txt

# Run scripts with full path
eeg_mel_env\Scripts\python.exe train.py
```

### Requirements

See [requirements.txt](requirements.txt) for full list. Key packages:

| Package | Version | Purpose |
|---|---|---|
| torch | ≥ 2.4.1+cu118 | Model training (GPU) |
| torchaudio | ≥ 2.4.1 | Audio processing |
| mne | ≥ 1.2 | EEG loading & filtering |
| librosa | ≥ 0.11 | Mel-spectrogram extraction |
| scipy | ≥ 1.10 | `.mat` file loading |
| einops | ≥ 0.8 | Tensor rearrangement |
| matplotlib | ≥ 3.7 | Visualization |
| speechbrain | ≥ 1.0 | HiFi-GAN vocoder (inference) |

---

## Usage

### 1. Train

```bash
# Mamba model (default)
python train.py

# Transformer + S4 model
# Set arch = "transformer_s4" in train.py CFG, then:
python train.py
```

Training logs are printed every 10 epochs. Best checkpoint saved to `checkpoints/best_model_mamba.pt`.

### 2. Evaluate

```bash
python evaluate.py
```

Outputs per-trial metrics (MSE, L1, Spectral Convergence, MCD, Cosine Similarity) and mel-spectrogram comparison plots to `results/`.

### 3. Inference (EEG → Speech)

```bash
# All trials, HiFi-GAN vocoder
python inference.py

# First 10 trials only, Griffin-Lim (no download required)
python inference.py --n 10 --vocoder griffinlim
```

Generated audio saved to `outputs/wav_pred/`, ground-truth audio to `outputs/wav_gt/`.

---

## Model Architecture

### Model 1: Transformer + S4-UNet

```
EEG (B, 62, 4000)
  │
  ▼  EEGSpatialEmbed — Conv1D(62→96, stride=10)   [patch_size=10 → 400 frames]
  │
  ▼  Transformer Encoder × 3
  │    └─ Multi-Head Attention (4 heads) + FFN (96→192→96) + LayerNorm
  │
  ▼  S4-UNet Decoder
  │    Encoder: 96→256 → 128(↓½) → 64(↓½)
  │    Bottleneck: S4Layer (diagonal complex SSM, FFT conv)
  │    Decoder: (64+128)→128(↑2) → (128+256)→256(↑2) → (256+96)→256
  │
  ▼  Linear(256→80) + Sigmoid
     Mel-spectrogram (B, 80, T)
```

| Component | Detail |
|---|---|
| Parameters | **1.33M** |
| EEG patch size | 10 ms (stride 10) |
| Transformer heads | 4 |
| S4 state dim | 16 |
| Dropout | 0.25 |

---

### Model 2: Mamba (S6) — Bidirectional

```
EEG (B, 62, 4000)
  │
  ▼  Patch Embed — Conv1D(62→96, stride=10) + GELU   [→ 400 frames]
  │
  ▼  Bidirectional Mamba Encoder × 4
  │    └─ Forward Mamba + Backward Mamba → concat → Linear
  │         MambaBlock: in_proj → depthwise conv → selective SSM (S6) → gate
  │         Selective SSM: B, C, Δ = f(x)  ← input-dependent
  │
  ▼  Mamba-UNet Decoder
  │    Encoder: 96→256 → 128(↓½) → 64(↓½)
  │    Bottleneck: MambaBlock
  │    Decoder: (64+128)→128(↑2) → (128+256)→256(↑2) → (256+96)→256
  │
  ▼  Linear(256→80) + Sigmoid
     Mel-spectrogram (B, 80, T)
```

| Component | Detail |
|---|---|
| Parameters | **3.75M** |
| SSM type | S6 (selective, input-dependent B/C/Δ) |
| Scan direction | Bidirectional (forward + backward) |
| d_inner | 2 × d_model (expand=2) |
| d_conv | 4 (depthwise causal conv) |
| d_state | 16 |

---

### S4 vs Mamba Comparison

| | S4 | Mamba (S6) |
|---|---|---|
| A, B, C | Fixed | **Input-dependent** B, C, Δ |
| Mechanism | Global convolution (FFT) | Selective state space + parallel scan |
| Sequence complexity | O(L log L) | O(L) |
| Memory | O(L) | O(L) |
| EEG suitability | Good | **Better** (selective attention to relevant time points) |

---

## Training Configuration

```python
CFG = dict(
    eeg_duration    = 4.0,        # EEG window (seconds)
    mel_time_frames = 128,        # fixed mel output frames
    n_mels          = 80,
    use_speaking    = True,       # include pronounced speech EEG (~2× data)
    augment         = True,       # noise / time-shift / channel-dropout / scaling
    d_model         = 96,
    n_enc_layers    = 3,          # (4 for Mamba)
    patch_size      = 10,         # ms
    d_state         = 16,
    dropout         = 0.25,
    epochs          = 500,
    batch_size      = 16,
    lr              = 3e-4,
    weight_decay    = 1e-3,
)
```

### Loss Function

```
L_total = L1 + 0.5 × SpectralConvergence + MultiResolutionSpectral
```

- **L1**: Frame-level reconstruction
- **Spectral Convergence**: `||target - pred||_F / ||target||_F`
- **Multi-Resolution Spectral**: L1 + log-L1 at temporal scales ×1, ×2, ×4

### EEG Data Augmentation

| Augmentation | Probability | Detail |
|---|---|---|
| Gaussian noise | 50% | σ = 0.05 × signal std |
| Time shift | 50% | ±100 ms |
| Channel dropout | 50% | 5% of channels zeroed |
| Amplitude scaling | 50% | ×[0.8, 1.2] |

---

## File Structure

```
.
├── data_loader.py    # KaraOne dataset loader + EEG augmentation
├── model.py          # Transformer encoder + S4-UNet decoder
├── model_mamba.py    # Bidirectional Mamba encoder + Mamba-UNet decoder
├── train.py          # Training loop (supports both architectures)
├── evaluate.py       # Metrics: MSE, L1, SC, MCD, CosSim
├── inference.py      # EEG → mel → audio (.wav) via HiFi-GAN or Griffin-Lim
├── requirements.txt
└── README.md
```

---

## Citation

If you use this code, please cite the KaraOne dataset:

```bibtex
@inproceedings{zhao2015classifying,
  title={Classifying phonological categories in imagined and articulated speech},
  author={Zhao, Shunan and Rudzicz, Frank},
  booktitle={ICASSP},
  year={2015}
}
```
