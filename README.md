# AudioFeatureExtractor - MFCC Calculation

A command-line application for calculating MFCC values (Mel-Frequency Cepstral Coefficients) from audio files.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Process a single file:
```bash
python mfcc_extractor.py path/to/audio.mp3
```

### Process a directory:
```bash
python mfcc_extractor.py path/to/directory
```

### Output to file:
```bash
python mfcc_extractor.py path/to/audio.mp3 -o results.json
```

## Output Format

The results are output in JSON format:
```json
[
  {
    "filename": "audio.mp3",
    "mfcc_means": [0.123, -0.456, ...]
  }
]
```

## Supported Formats
- MP3
- FLAC 