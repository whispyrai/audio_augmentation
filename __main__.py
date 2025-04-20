import os
import argparse
import random
from pathlib import Path
from torch_audiomentations import Compose, Gain, PitchShift, AddBackgroundNoise
import torchaudio
import torch

# torchaudio.set_audio_backend("soundfile")
SAMPLE_RATE = 48000

noise_files = []
noise_dir = Path("audio_augmentation/noise").resolve()
if noise_dir.exists():
    temp = list(noise_dir.rglob("*.wav"))
    noise_files = [str(file) for file in temp]

AUGMENTATIONS = {
    "pitch": PitchShift(
        min_transpose_semitones=-0.5,
        max_transpose_semitones=0.5,
        sample_rate=SAMPLE_RATE,
        output_type="tensor",
    ),
    "gain": Gain(
        min_gain_in_db=-3.0,
        max_gain_in_db=3.0,
        p=1.0,
        output_type="tensor",
    ),
}

if len(noise_files) > 0:
    AUGMENTATIONS["noise"] = AddBackgroundNoise(
        background_paths=noise_files,
        min_snr_in_db=15,
        max_snr_in_db=30,
        p=1.0,
        output_type="tensor",
    )


def randomly_select_augmentations():
    num_augs = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2], k=1)[0]
    return random.sample(list(AUGMENTATIONS.items()), num_augs)


def augment_chunk(audio, augmentations):
    augmented_audio = audio.clone().unsqueeze(0)
    for _, aug in augmentations:
        augmented_audio = aug(augmented_audio, sample_rate=SAMPLE_RATE)
    return augmented_audio.squeeze(0)


def process_directory(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in input_dir.glob("*.wav"):
        base_name = audio_file.stem
        lab_file = input_dir / f"{base_name}.lab"

        audio, sr = torchaudio.load(audio_file)

        # Save original files
        torchaudio.save(output_dir / f"{base_name}.wav", audio, sr)
        lab_out = output_dir / f"{base_name}.lab"
        lab_out.write_text(lab_file.read_text())

        augmentations = randomly_select_augmentations()

        if augmentations:
            aug_names = "_".join([name for name, _ in augmentations])
            print(aug_names)
            augmented_audio = augment_chunk(audio, augmentations)
            aug_audio_path = output_dir / f"{base_name}_{aug_names}.wav"
            torchaudio.save(aug_audio_path, augmented_audio, SAMPLE_RATE)

            # Copy transcription for augmented chunk
            aug_lab_path = output_dir / f"{base_name}_{aug_names}.lab"
            aug_lab_path.write_text(lab_file.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio Augmentation for Fish Speech fine-tuning"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Relative path to the input directory containing audio chunks",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir).resolve()
    output_path = input_path.parent / "augmented"

    process_directory(input_path, output_path)

    print(f"Augmentation complete. Files saved in: {output_path}")
