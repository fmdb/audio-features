import os
import librosa
import numpy as np
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Union, Optional
import typer
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

BUILD_ID = os.getenv('BUILD_ID', 'development')

# Logging-Setup
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)

app = typer.Typer()
thread_local = threading.local()

def calculate_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def extract_metadata(audio_path: str, file_number: int) -> Dict:
    file_path = Path(audio_path)
    file_size = file_path.stat().st_size / (1024 * 1024)
    
    metadata = {
        "filename": file_path.name,
        "file_number": file_number,
        "file_size_in_mb": round(file_size, 2),
        "lossless": file_path.suffix.lower() == '.flac',
        "sha256": calculate_sha256(audio_path),
        "build_id": BUILD_ID
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

def calculate_mfcc(audio_path: str, file_number: int) -> Dict:
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
    logger.info(f"Processing file: {audio_path}")
    
    y, sr = librosa.load(audio_path, sr=None)
    logger.info(f"Sample rate: {sr} Hz")
    
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
    
    result = extract_metadata(audio_path, file_number)
    result.update({
        "features": {
            "mfcc": mfcc.mean(axis=1).tolist()
        }
    })
    
    return result

def process_audio_files(input_path: Union[str, Path], output: Optional[Path] = None) -> List[Dict]:
    input_path = Path(input_path)
    logger.info(f"Start processing: {input_path}")
    results = []
    
    if not input_path.exists():
        raise typer.BadParameter(f"Input path {input_path} does not exist")

    def process_file(args: tuple) -> Dict:
        file_path, file_number = args
        try:
            result = calculate_mfcc(str(file_path), file_number)
            if result:
                results.append(result)
                if not output:
                    print(json.dumps(result, indent=2))
            return result
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None

    if input_path.is_file():
        files = [(input_path, 1)]
    else:
        files = [(f, i+1) for i, f in enumerate(sorted([f for f in input_path.glob('*') if f.suffix.lower() in ['.mp3', '.flac']]))]

    max_workers = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(process_file, files))

    logger.info(f"Processing completed. {len(results)} files processed.")

    # Sort results by file_number
    results.sort(key=lambda x: x['metadata']['file_number'])

    if output:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results

@app.command()
def main(
    input_path: Path = typer.Argument(..., exists=True, help="Input audio file or directory"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for JSON-Results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    if verbose:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
        for handler in logger.handlers:
            handler.setFormatter(formatter)
    
    process_audio_files(input_path, output)

if __name__ == "__main__":
    app()