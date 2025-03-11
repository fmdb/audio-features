from setuptools import setup, find_packages

setup(
    name="fmdb-mfcc-extractor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "librosa==0.11.0",
        "numpy>=1.20.0",
        "soundfile>=0.12.1",
        "pydub>=0.25.1"
    ],
    entry_points={
        'console_scripts': [
            'mfcc-extractor=mfcc_extractor:main',
        ],
    },
    python_requires='>=3.8'
) 