# AudioFeatureExtractor - MFCC Calculation

A command-line application for calculating MFCC values ([Mel-Frequency Cepstral Coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)) from audio files.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Process a single file:
```bash
python audio_mfcc/app.py path/to/audio.mp3
```

### Process a directory:
```bash
python audio_mfcc/app.py path/to/directory
```

### Output to file:
```bash
python audio_mfcc/app.py path/to/audio.mp3 -o results.json
```

### Enable verbose logging:
```bash
python audio_mfcc/app.py path/to/audio.mp3 --verbose
```

## Output Format

The results are output in JSON format. A complete JSON schema is available in `audio_mfcc/schema.json`.

Example output:
```json
[
  {
    "metadata": {
      "filename": "audio.mp3",
      "file_number": 1,
      "file_size_in_mb": 5.67,
      "lossless": false,
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "title": "Song Title",
      "artist": "Artist Name",
      "album": "Album Name",
      "year": "2024",
      "genre": "Rock",
      "isrc": "USRC17607839",
      "duration_in_ms": 180000,
      "bitrate": 320,
      "sample_rate": 44100,
      "channels": "Stereo"
    },
    "features": {
      "mfcc": [0.123, -0.456, ...]
    }
  }
]
```

## Supported Formats
- MP3
- FLAC 

## Disclaimer

This project is experimental.
