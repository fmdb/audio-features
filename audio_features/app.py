import os
import librosa
import numpy as np
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import typer
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from tqdm import tqdm
import pickle
import tempfile

BUILD_ID = os.getenv('BUILD_ID', 'development')

# Logging setup
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)

app = typer.Typer()
thread_local = threading.local()

class AudioProcessor:
    """Main class for audio file processing."""
    
    def __init__(self, cache_dir: Optional[Path] = None, use_cache: bool = True):
        """
        Initializes the AudioProcessor.
        
        Args:
            cache_dir: Directory for feature cache
            use_cache: Whether to use caching
        """
        self.use_cache = use_cache
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "fmdb_audio_features_cache"
        if self.use_cache and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory created: {self.cache_dir}")
    
    def calculate_sha256(self, file_path: str) -> str:
        """Calculates the SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def get_cache_path(self, audio_path: str, file_hash: str) -> Path:
        """Determines the path for the cache file."""
        return self.cache_dir / f"{file_hash}.pickle"
    
    def extract_metadata(self, audio_path: str, file_number: int) -> Dict:
        """Extracts metadata from an audio file."""
        file_path = Path(audio_path)
        file_size = file_path.stat().st_size / (1024 * 1024)
        file_hash = self.calculate_sha256(audio_path)
        
        metadata = {
            "filename": file_path.name,
            "file_number": file_number,
            "file_size_in_mb": round(file_size, 2),
            "lossless": file_path.suffix.lower() == '.flac',
            "sha256": file_hash,
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
        
        return metadata
    
    def check_cache(self, audio_path: str) -> Optional[Dict]:
        """Checks if a cache file exists and returns the features if available."""
        if not self.use_cache:
            return None
            
        file_hash = self.calculate_sha256(audio_path)
        cache_path = self.get_cache_path(audio_path, file_hash)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                if cached_data.get('build_id') == BUILD_ID:
                    logger.info(f"Cache hit for {audio_path}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_path}: {e}")
        
        return None
    
    def save_to_cache(self, audio_path: str, result: Dict) -> None:
        """Saves results to cache."""
        if not self.use_cache:
            return
            
        file_hash = self.calculate_sha256(audio_path)
        cache_path = self.get_cache_path(audio_path, file_hash)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            logger.info(f"Cache saved for {audio_path}")
        except Exception as e:
            logger.warning(f"Error saving cache file {cache_path}: {e}")
    
    def calculate_audio_features(self, audio_path: str, file_number: int) -> Dict:
        """
        Calculates various audio features including MFCC, spectral contrast,
        chroma, and tempo.
        """
        logger.info(f"Processing file: {audio_path}")
        
        # Check if in cache
        cached_result = self.check_cache(audio_path)
        if cached_result:
            cached_result['metadata']['file_number'] = file_number  # Update file number
            return cached_result
        
        # Extract metadata
        metadata = self.extract_metadata(audio_path, file_number)
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            logger.info(f"Sample rate: {sr} Hz")
            
            # Calculate Mel spectrogram
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
            
            # Calculate MFCC
            mfcc = librosa.feature.mfcc(
                S=log_mel_spec,
                n_mfcc=13
            )
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=y, 
                sr=sr,
                n_fft=2048,
                hop_length=512
            )
            
            # Chroma
            chroma = librosa.feature.chroma_stft(
                y=y, 
                sr=sr,
                n_fft=2048,
                hop_length=512
            )
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Assemble result
            result = {
                "metadata": metadata,
                "features": {
                    "mfcc": mfcc.mean(axis=1).tolist(),
                    "spectral_contrast": spectral_contrast.mean(axis=1).tolist(),
                    "chroma": chroma.mean(axis=1).tolist(),
                    "tempo": float(tempo)
                }
            }
            
            # Save to cache
            self.save_to_cache(audio_path, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            # Ensure that at least metadata is returned
            return {"metadata": metadata, "error": str(e)}
    
    def process_audio_files(self, input_path: Union[str, Path], output: Optional[Path] = None, verbose: bool = False) -> List[Dict]:
        """
        Processes a single audio file or all audio files in a directory.
        
        Args:
            input_path: Path to audio file or directory
            output: Optional output path for JSON results
            verbose: Whether to display verbose output
            
        Returns:
            List of processing results
        """
        input_path = Path(input_path)
        logger.info(f"Starting processing: {input_path}")
        
        if not input_path.exists():
            raise typer.BadParameter(f"Input path {input_path} does not exist")

        # Lock for thread-safe result storage
        results_lock = threading.Lock()
        results = []
        errors = []

        def process_file(args: tuple) -> Optional[Dict]:
            file_path, file_number = args
            try:
                result = self.calculate_audio_features(str(file_path), file_number)
                if result:
                    with results_lock:
                        if "error" in result:
                            errors.append((file_path, result["error"]))
                        else:
                            results.append(result)
                    if not output:
                        if verbose:
                            # Display JSON output when in verbose mode and no output file is specified
                            print(json.dumps(result, indent=2))
                        else:
                            logger.info(f"Processing successful: {file_path}")
                return result
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                logger.error(error_msg)
                with results_lock:
                    errors.append((file_path, str(e)))
                return None

        if input_path.is_file():
            files = [(input_path, 1)]
        else:
            files = [(f, i+1) for i, f in enumerate(sorted([f for f in input_path.glob('*') if f.suffix.lower() in ['.mp3', '.flac']]))]

        max_workers = min(os.cpu_count() or 4, len(files))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Progress display with tqdm
            futures = [executor.submit(process_file, args) for args in files]
            
            with tqdm(total=len(files), desc="Processing audio files") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Unexpected error in thread: {str(e)}")
                    finally:
                        pbar.update(1)

        # Show error summary
        if errors:
            logger.warning(f"{len(errors)} files could not be processed:")
            for file_path, error in errors:
                logger.warning(f"  - {file_path}: {error}")

        logger.info(f"Processing completed. {len(results)} files successfully processed.")

        # Sort results by file_number
        results.sort(key=lambda x: x['metadata']['file_number'])

        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to: {output}")

        return results

@app.command()
def main(
    input_path: Path = typer.Argument(..., exists=True, help="Input audio file or directory"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for JSON results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable feature cache usage"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Directory for feature cache")
):
    """
    FMDB Audio Features: Extracts audio features from MP3 or FLAC files.
    
    This application can process a single audio file or all audio files in a directory
    and calculates various audio features (MFCC, spectral contrast, chroma, tempo).
    Results can be saved as a JSON file.
    """
    # Configure logging
    if verbose:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
        for handler in logger.handlers:
            handler.setFormatter(formatter)
    
    # Initialize audio processor
    processor = AudioProcessor(cache_dir=cache_dir, use_cache=not no_cache)
    
    try:
        # Start processing
        start_time = time.time()
        results = processor.process_audio_files(input_path, output, verbose)
        elapsed_time = time.time() - start_time
        
        # Show summary
        print(f"\nProcessing completed.")
        print(f"Processed files: {len(results)}")
        print(f"Duration: {elapsed_time:.2f} seconds")
        
        if output:
            print(f"Results saved to: {output}")
        
        return results
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app() 