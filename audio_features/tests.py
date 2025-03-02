import unittest
import tempfile
import os
import json
from pathlib import Path
from audio_features.app import AudioProcessor

class TestAudioProcessor(unittest.TestCase):
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.processor = AudioProcessor(cache_dir=self.cache_dir)
        
    def tearDown(self):
        """Clean up the test environment."""
        self.temp_dir.cleanup()
    
    def test_sha256_calculation(self):
        """Test for SHA256 hash calculation."""
        # Create a test file
        test_file_path = Path(self.temp_dir.name) / "test_file.txt"
        with open(test_file_path, 'w') as f:
            f.write("Test content")
        
        # Calculate expected hash
        import hashlib
        expected_hash = hashlib.sha256(b"Test content").hexdigest()
        
        # Test function
        calculated_hash = self.processor.calculate_sha256(str(test_file_path))
        self.assertEqual(calculated_hash, expected_hash)
    
    def test_cache_functionality(self):
        """Test for cache functionality."""
        # Create a test file
        test_file_path = Path(self.temp_dir.name) / "test_audio.mp3"
        with open(test_file_path, 'wb') as f:
            f.write(b"Dummy MP3 content")
            
        # Patch the check_cache method to provide consistent return
        original_check_cache = self.processor.check_cache
        
        def mock_check_cache(audio_path):
            if audio_path == str(test_file_path):
                return dummy_result
            return original_check_cache(audio_path)
            
        # Create a dummy result file with correct build ID
        dummy_result = {
            "metadata": {
                "filename": test_file_path.name,
                "file_number": 1,
                "sha256": self.processor.calculate_sha256(str(test_file_path)),
                "build_id": self.processor.calculate_sha256(str(test_file_path))  # Use hash as build ID
            },
            "features": {"mfcc": [1, 2, 3]}
        }
        
        # Save to cache
        self.processor.save_to_cache(str(test_file_path), dummy_result)
        
        # Check if file was created in cache directory
        cache_files = list(self.cache_dir.glob("*.pickle"))
        self.assertTrue(len(cache_files) > 0, "No cache file created")
        
        # Check if cache file has correct content
        with open(cache_files[0], 'rb') as f:
            import pickle
            cached_data = pickle.load(f)
            self.assertEqual(cached_data["features"]["mfcc"], [1, 2, 3])
    
if __name__ == "__main__":
    unittest.main() 