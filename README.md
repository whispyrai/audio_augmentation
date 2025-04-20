# Audio Augmentation Toolkit

*A lightweight, self‑contained utility for preparing training data for Fish Speech fine‑tuning at **Whispyr AI**.*

---

## 1  Why this exists
Fine‑tuning Fish Speech (or any ASR/TTS model) benefits enormously from acoustic diversity. This repository automates two repetitive steps:

1. **Fetching a curated library of open‑licensed noise WAVs** from an S3 bucket.
2. **Creating pitch‑shifted, gain‑adjusted and/or noise‑mixed copies** of every labeled chunk in your dataset, while keeping `.lab` transcriptions in sync.

The result is an *augmented* directory that can be fed directly to Fish Speech’s training CLI with no additional work.

---

## 2  Repository layout
```text
├── audio_augmentation
│   ├── __init__.py       # (empty – marks the module)
│   ├── __main__.py       # entry‑point for augmentation
│   └── download_noise.py # one‑off noise fetcher
│   └── README.md         # you are here
```

---

## 3  Quick start
```bash
# 0)  Install Python 3.10+ and ffmpeg on your machine first

# 1)  Clone and create an isolated env
$ git clone git@github.com:whispyr‑ai/audio_augmentation.git

# 2)  Install core dependencies
$ pip install -r audio_augmentation/requirements.txt  # or see 4 for the list

# 3)  Download the noise corpus (≈500 MB, one time only)
$ python audio_augmentation/download_noise

# 4)  Augment a folder of labelled chunks
$ python audio_augmentation /path/to/my_chunks
```
> **Important:** always execute the modules from *one directory level **above*** the package, i.e. `python -m audio_augmentation ...`. Running them from inside the folder changes the import path and will break things.

---

## 4  Dependencies
| Package | Reason |
|---------|--------|
| **torch >= 2.1** & **torchaudio** | tensor ops & I/O |
| **torch‑audiomentations >= 0.13** | high‑performance audio transforms |
| **requests** & **beautifulsoup4** | scraping the S3 bucket listing |
| **tqdm** *(optional)* | neat progress bars |
| **soundfile** *(optional)* | alternate backend for torchaudio |

Install manually:
```bash
pip install torch torchaudio torch-audiomentations requests beautifulsoup4 tqdm soundfile
```
GPU builds of PyTorch are recommended if you are processing tens of hours of audio.

---

## 5  Using the augmentation CLI
### 5.1  Required input structure
The script expects a *flat* directory where each audio chunk has a matching transcription:
```text
my_chunks/
├── chunk_001.wav
├── chunk_001.lab
├── chunk_002.wav
├── chunk_002.lab
└── ...
```
The `.lab` file must be **plain UTF‑8 text** containing the transcription of the WAV.

### 5.2  Command‑line arguments
| Positional | Description |
|------------|-------------|
| `input_dir` | Path to folder described above |

There are no optional flags yet – the goal is zero‑config.

### 5.3  What happens under the hood
1. All `.wav`/`.lab` pairs are copied verbatim into a sibling folder called **`augmented/`**.
2. For each original WAV we sample *N* ∈ {0, 1, 2} augmentations with probabilities 0.3 / 0.5 / 0.2, respectively.
3. Supported transforms:
   * **PitchShift**: ±0.5 semitones (@48 kHz).
   * **Gain**: ‑3 to +3 dB.
   * **AddBackgroundNoise**: mixes a random noise file at 15–30 dB SNR (only if the noise corpus was downloaded).
4. Augmented files are saved as `<stem>_<aug1>[_<aug2>].wav`, e.g. `chunk_001_pitch_noise.wav`. A matching `.lab` copy is written with the same stem.

### 5.4  Output example
```text
augmented/
├── chunk_001.wav
├── chunk_001.lab
├── chunk_001_pitch_noise.wav
├── chunk_001_pitch_noise.lab
└── ...
```
You can now point Fish Speech’s training script at `augmented/`.

---

## 6  Noise corpus details
`download_noise.py` crawls `https://whispyr-noise-files.s3.amazonaws.com/` and downloads every `.wav` it finds. Files are cached locally under `audio_augmentation/noise/` and skipped on subsequent runs.

*If you want to add your own noises, just drop them anywhere inside that folder – they will be picked up automatically.*

---

## 7  Troubleshooting
| Symptom | Likely cause / fix |
|---------|-------------------|
| `ModuleNotFoundError: torch_audiomentations` | Activate your venv or reinstall dependencies. |
| `RuntimeError: No audio backend` | Install `soundfile` or ensure ffmpeg is in PATH. |
| Augmented files sound clipped | Lower `max_gain_in_db` in `__main__.py`. |
| No noise added | You forgot to run `download_noise.py`, or the noise folder is empty. |

