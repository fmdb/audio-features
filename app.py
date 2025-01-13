import click
import librosa
import json
import os
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""Calculate MFCCs using parameters aligned with Essentia implementation.

Parameters:
- Sampling rate: 44100 Hz
- FFT size: 2048
- Window size: 2048
- Hop length: 512
- Window type: Hann
- Number of MFCCs: 13
- Number of Mel bands: 40
- Min frequency: 0 Hz
- Max frequency: Nyquist frequency (sr/2)
- Power: Energy (2.0)
"""
def calculate_mfcc(audio_path):
    # Load audio with specific sampling rate
    y, sr = librosa.load(audio_path, sr=44100)

    # Calculate mel spectrogram with specified parameters
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=4096,
        win_length=4096,
        hop_length=512,
        window='hann',
        n_mels=40,
        fmin=0.0,
        fmax=sr/2,
        power=2.0
    )

    # Convert to log mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Calculate MFCCs
    mfcc = librosa.feature.mfcc(
        S=log_mel_spec,  # Use pre-computed log mel spectrogram
        n_mfcc=13
    )

    return {
        "filename": os.path.basename(audio_path),
        "mfcc_means": mfcc.mean(axis=1).tolist()
    }

def process_audio_files(input_path):
    logging.info(f"Start processing: {input_path}")
    results = []
    input_path = Path(input_path)
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.mp3', '.flac']:
            results.append(calculate_mfcc(str(input_path)))
    else:
        audio_files = sorted([f for f in input_path.glob('*') if f.suffix.lower() in ['.mp3', '.flac']])
        for file_path in audio_files:
            results.append(calculate_mfcc(str(file_path)))
    
    logging.info(f"Processing completed. {len(results)} files processed.")
    return results

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for JSON-Results')
def main(input_path, output):
    results = process_audio_files(input_path)
    json_output = json.dumps(results, indent=2, ensure_ascii=False)
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(json_output)
    else:
        click.echo(json_output)

if __name__ == '__main__':
    main() 