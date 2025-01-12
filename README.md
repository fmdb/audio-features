# AudioFeatureExtractor - MFCC Berechnung

Eine Kommandozeilenanwendung zur Berechnung von MFCC-Werten (Mel-Frequency Cepstral Coefficients) aus Audiodateien.

## Installation

```bash
pip install -r requirements.txt
```

## Verwendung

### Einzelne Datei verarbeiten:
```bash
python mfcc_extractor.py pfad/zur/audio.mp3
```

### Verzeichnis verarbeiten:
```bash
python mfcc_extractor.py pfad/zum/verzeichnis
```

### Mit Ausgabe in Datei:
```bash
python mfcc_extractor.py pfad/zur/audio.mp3 -o ergebnisse.json
```

## Ausgabeformat

Die Ergebnisse werden im JSON-Format ausgegeben:
```json
[
  {
    "filename": "audio.mp3",
    "mfcc_means": [0.123, -0.456, ...]
  }
]
```

## Unterstützte Formate
- MP3
- FLAC 