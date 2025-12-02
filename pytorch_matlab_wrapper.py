import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import os

class SmileFrownDetector:
    def __init__(self, model_path="smile_frown_resnet18.pth", 
                 class_mapping_path="class_mapping.json"):
        """
        Initialize the smile/frown detector
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load class mapping
        if os.path.exists(class_mapping_path):
            with open(class_mapping_path, "r") as f:
                class_to_idx = json.load(f)
            self.labels = [None] * len(class_to_idx)
            for class_name, idx in class_to_idx.items():
                self.labels[idx] = class_name
        else:
            self.labels = ["frown", "smile"]
        
        # Preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded on {self.device}")
        print(f"Labels: {self.labels}")
    
    def predict_from_array(self, image_array):
        """
        Predict from a numpy array (H x W x 3, RGB, uint8)
        
        Args:
            image_array: numpy array of shape (H, W, 3) with RGB values 0-255
            
        Returns:
            dict with 'label', 'confidence', 'probabilities'
        """
        # Convert numpy array to PIL Image
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        face_pil = Image.fromarray(image_array)
        
        # Preprocess
        img_tensor = self.preprocess(face_pil).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            preds = self.model(img_tensor)
            probs = torch.softmax(preds, dim=1)[0]
            idx = torch.argmax(preds, 1).item()
            label = self.labels[idx]
            conf = probs[idx].item()
        
        return {
            'label': label,
            'confidence': float(conf),
            'probabilities': probs.cpu().numpy().tolist(),
            'class_index': int(idx)
        }


# Global detector instance
_detector = None

def initialize(model_path="smile_frown_resnet18.pth", 
               class_mapping_path="class_mapping.json"):
    """Initialize the detector (call once from MATLAB)"""
    global _detector
    _detector = SmileFrownDetector(model_path, class_mapping_path)
    return True

def predict(image_array):
    """
    Run inference on a single image
    
    Args:
        image_array: numpy array (H, W, 3) RGB uint8
        
    Returns:
        prediction dict
    """
    global _detector
    if _detector is None:
        raise RuntimeError("Detector not initialized. Call initialize() first.")
    return _detector.predict_from_array(image_array)

def get_labels():
    """Get the class labels"""
    global _detector
    if _detector is None:
        raise RuntimeError("Detector not initialized.")
    return _detector.labels