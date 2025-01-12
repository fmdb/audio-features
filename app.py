import click
import librosa
import json
import os
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_mfcc(audio_path):
    logging.info(f"Processing file: {audio_path}")
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1).tolist()
    return {
        "filename": os.path.basename(audio_path),
        "mfcc_means": mfcc_means
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