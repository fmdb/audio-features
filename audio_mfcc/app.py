import librosa
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Union, Optional
import typer
from mutagen.mp3 import MP3
from mutagen.flac import FLAC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = typer.Typer()

def extract_metadata(audio_path: str) -> Dict:
    file_path = Path(audio_path)
    file_size = file_path.stat().st_size / (1024 * 1024)
    
    metadata = {
        "filename": file_path.name,
        "file_size_in_mb": round(file_size, 2),
        "lossless": file_path.suffix.lower() == '.flac'
    }
    
    if file_path.suffix.lower() == '.mp3':
        mp3 = MP3(audio_path)
        metadata.update({
            "title": mp3.tags.get('TIT2', [''])[0] if mp3.tags else '',
            "artist": mp3.tags.get('TPE1', [''])[0] if mp3.tags else '',
            "album": mp3.tags.get('TALB', [''])[0] if mp3.tags else '',
            "year": mp3.tags.get('TDRC', [''])[0] if mp3.tags else '',
            "genre": mp3.tags.get('TCON', [''])[0] if mp3.tags else '',
            "isrc": mp3.tags.get('TSRC', [''])[0] if mp3.tags and 'TSRC' in mp3.tags else '',
            "duration_in_ms": int(mp3.info.length * 1000),
            "bitrate": int(mp3.info.bitrate / 1000),
            "sample_rate": mp3.info.sample_rate,
            "channels": "Stereo" if mp3.info.channels == 2 else "Mono"
        })
    elif file_path.suffix.lower() == '.flac':
        flac = FLAC(audio_path)
        metadata.update({
            "title": flac.tags.get('title', [''])[0] if flac.tags else '',
            "artist": flac.tags.get('artist', [''])[0] if flac.tags else '',
            "album": flac.tags.get('album', [''])[0] if flac.tags else '',
            "year": flac.tags.get('date', [''])[0] if flac.tags else '',
            "genre": flac.tags.get('genre', [''])[0] if flac.tags else '',
            "isrc": flac.tags.get('isrc', [''])[0] if flac.tags and 'isrc' in flac.tags else '',
            "duration_in_ms": int(flac.info.length * 1000),
            "bitrate": int(flac.info.bitrate / 1000),
            "sample_rate": flac.info.sample_rate,
            "channels": "Stereo" if flac.info.channels == 2 else "Mono"
        })
    
    return { "metadata": metadata }

def calculate_mfcc(audio_path: str) -> Dict:
    """Calculate MFCCs using parameters aligned with Essentia implementation.
    
    Parameters:
    - Sampling rate: Original sample rate of the file
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
    logging.info(f"Processing file: {audio_path}")
    
    y, sr = librosa.load(audio_path, sr=None)
    logging.info(f"Sample rate: {sr} Hz")
    
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        window='hann',
        n_mels=40,
        fmin=0.0,
        fmax=sr/2,
        power=2.0
    )
    
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    mfcc = librosa.feature.mfcc(
        S=log_mel_spec,
        n_mfcc=13
    )
    
    result = extract_metadata(audio_path)
    result.update({
        "features": {
            "mfcc": mfcc.mean(axis=1).tolist()
        }
    })
    
    return result

def process_audio_files(input_path: Union[str, Path]) -> List[Dict]:
    input_path = Path(input_path)
    logging.info(f"Start processing: {input_path}")
    results = []

    if not input_path.exists():
        raise typer.BadParameter(f"Input path {input_path} does not exist")

    if input_path.is_file():
        results.append(calculate_mfcc(str(input_path)))
    else:
        audio_files = sorted([f for f in input_path.glob('*') if f.suffix.lower() in ['.mp3', '.flac']])
        for file_path in audio_files:
            results.append(calculate_mfcc(str(file_path)))

    logging.info(f"Processing completed. {len(results)} files processed.")
    return results

@app.command()
def main(
    input_path: Path = typer.Argument(..., exists=True, help="Input audio file or directory"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for JSON-Results")
):
    results = process_audio_files(input_path)
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    app()