import unittest
import numpy as np
import torch
import cv2
import os
import sys
from unittest.mock import patch

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.preprocessing.preprocess_images import ImagePreprocessor
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.models.model import GestureClassifier

class TestGestureRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.preprocessor = ImagePreprocessor()
        cls.extractor = FeatureExtractor()
        cls.input_size = 16384  # Updated to match the model's expected input size
        cls.classifier = GestureClassifier(input_size=cls.input_size)
        
        # Load the model if it exists
        model_path = os.path.join(parent_dir, 'models', 'saved', 'best_model')
        if os.path.exists(f"{model_path}.pt"):
            cls.classifier.load_model(model_path)

    def test_preprocessor(self):
        """Test if the preprocessor can handle different input types"""
        # Test with a random image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mock the detect_hand method to return a valid bbox
        with patch.object(self.preprocessor, 'detect_hand', return_value=(50, 50, 200, 200)):
            processed = self.preprocessor.preprocess_image(test_image)
            self.assertIsNotNone(processed)
            self.assertEqual(len(processed.shape), 2)  # Should be grayscale

    def test_feature_extractor(self):
        """Test if the feature extractor can process preprocessed images"""
        # Create a sample preprocessed image
        test_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        features = self.extractor.extract_all_features(test_image)
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)

    def test_model_prediction(self):
        """Test if the model can make predictions"""
        # Create sample features
        sample_features = np.random.rand(self.input_size).astype(np.float32)
        prediction = self.classifier.predict(np.array([sample_features]))[0]
        self.assertIsInstance(prediction, (int, np.integer))
        self.assertIn(prediction, [0, 1, 2])  # Should predict one of three gestures

    def test_model_probabilities(self):
        """Test if the model outputs valid probabilities"""
        sample_features = np.random.rand(self.input_size).astype(np.float32)
        features_tensor = torch.FloatTensor([sample_features]).to(self.classifier.device)
        with torch.no_grad():
            outputs = self.classifier(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        self.assertEqual(len(probabilities), 3)  # Three gesture classes
        self.assertAlmostEqual(np.sum(probabilities), 1.0)  # Probabilities should sum to 1
        self.assertTrue(np.all(probabilities >= 0) and np.all(probabilities <= 1))  # Valid probability range

if __name__ == '__main__':
    unittest.main() 