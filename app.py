import click
import librosa
import json
import os
from pathlib import Path
import numpy as np

def calculate_mfcc(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1).tolist()
    return {
        "filename": os.path.basename(audio_path),
        "mfcc_means": mfcc_means
    }

def process_audio_files(input_path):
    results = []
    input_path = Path(input_path)
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.mp3', '.flac']:
            results.append(calculate_mfcc(str(input_path)))
    else:
        for file_path in input_path.glob('*'):
            if file_path.suffix.lower() in ['.mp3', '.flac']:
                results.append(calculate_mfcc(str(file_path)))
    
    return results

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Outputfile for JSON-Results')
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