# FMDB Audio Features

A tool for extracting audio features from MP3 and FLAC files.

## Features

- Extraction of MFCCs (Mel-Frequency Cepstral Coefficients)
- Extraction of spectral contrast
- Extraction of chroma features
- Tempo detection
- Support for MP3 and FLAC
- Caching system for faster repeated analyses
- Parallel processing of multiple files
- Progress display
- Detailed metadata extraction

## Requirements

- Python 3.12, 3.13, or 3.14
- Dependencies listed in requirements.txt

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Simple usage with a single file
python -m audio_features.app input.mp3

# Output to a JSON file
python -m audio_features.app input.mp3 -o results.json

# Processing an entire directory
python -m audio_features.app music_folder/ -o results.json

# With detailed logs
python -m audio_features.app input.mp3 -v

# Without cache
python -m audio_features.app input.mp3 --no-cache

# With custom cache directory
python -m audio_features.app input.mp3 --cache-dir /tmp/my_cache
```

### Using as a Module

```python
from audio_features.app import AudioProcessor

# Initialize processor
processor = AudioProcessor()

# Process a single file
result = processor.calculate_audio_features("path/to/file.mp3", 1)

# Process multiple files
results = processor.process_audio_files("directory/with/files")
```

## Output Format

The output is a JSON document with the following format:

```json
[
  {
    "metadata": {
      "filename": "example.mp3",
      "file_number": 1,
      "file_size_in_mb": 5.28,
      "lossless": false,
      "sha256": "aef1c5...",
      "build_id": "development",
      "title": "Example Title",
      "artist": "Example Artist",
      "album": "Example Album",
      "year": "2023",
      "genre": "Electro",
      "isrc": "ABC123456789",
      "duration_in_ms": 180000,
      "bitrate": 320,
      "sample_rate": 44100,
      "channels": "Stereo"
    },
    "features": {
      "mfcc": [0.1, 0.2, ...],
      "spectral_contrast": [0.3, 0.4, ...],
      "chroma": [0.5, 0.6, ...],
      "tempo": 120.5
    }
  }
]
```

## Tests

```bash
python -m unittest audio_features.tests
```

## Supported Formats
- MP3
- FLAC 

## Disclaimer

This project is experimental.
