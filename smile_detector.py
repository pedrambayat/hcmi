import cv2
import torch
import json
import numpy as np
from torchvision import transforms
from PIL import Image
import time

class SmileDetector:
    def __init__(self, model_path='smile_frown_resnet18.pth', class_mapping_path='class_mapping.json'):
        """Initialize the smile detector with PyTorch model."""
        print('Loading PyTorch model...')
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        print(f'Class mapping loaded: {self.class_mapping}')
        
        # Handle different mapping formats
        # If mapping is like {"0": "smile", "1": "frown"}, keep as is
        # If mapping is like {"smile": 0, "frown": 1}, invert it
        if isinstance(list(self.class_mapping.values())[0], str):
            # Already in correct format: index -> label
            pass
        else:
            # Invert: label -> index to index -> label
            self.class_mapping = {str(v): k for k, v in self.class_mapping.items()}
            print(f'Inverted mapping to: {self.class_mapping}')
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model architecture (ResNet18 with custom output)
        from torchvision import models
        num_classes = len(self.class_mapping)
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print('Model loaded successfully!\n')
    
    def predict(self, face_img):
        """Run inference on a face image."""
        # Convert BGR to RGB if needed
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(face_img)
        
        # Preprocess
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get label
        label = self.class_mapping[str(predicted.item())]
        
        return {
            'label': label,
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy()[0]
        }

def expand_bbox(bbox, frame_shape, ratio=0.1):
    """Expand bounding box by ratio while staying within frame bounds."""
    x, y, w, h = bbox
    
    # Calculate expansion
    dx = w * ratio
    dy = h * ratio
    
    # Apply expansion
    new_x = max(0, int(x - dx))
    new_y = max(0, int(y - dy))
    new_w = min(frame_shape[1] - new_x, int(w + 2*dx))
    new_h = min(frame_shape[0] - new_y, int(h + 2*dy))
    
    return (new_x, new_y, new_w, new_h)

def realtime_smile_detector():
    """
    Returns: 'Approve' if majority smiling, 'Disapprove' if majority frowning
    """
    # Initialize detector
    detector = SmileDetector()
    
    # Setup webcam
    print('Initializing webcam...')
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")
    
    # Get camera resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Webcam initialized')
    print(f'Camera resolution: {frame_width} x {frame_height}')
    
    # Load face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Arrays to store predictions
    prediction_results = []
    capture_interval = 1.0  # Capture every 1 second
    last_capture_time = time.time()
    
    # Performance tracking
    start_time = time.time()
    capture_count = 0
    current_prediction = ''
    
    print('\n=== Starting Smile/Frown Assessment ===')
    print('Recording one sample per second...')
    print('Press SPACEBAR or Q to quit.\n')
    print(f'{"Sample #":<10} | {"Prediction":<12} | {"Conf.":<10} | {"Running Count":<15}')
    print('-' * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        # Check if it's time to capture and analyze
        current_time = time.time()
        should_capture = (current_time - last_capture_time) >= capture_interval
        
        if len(faces) > 0 and should_capture:
            # Use the largest face
            areas = [w * h for (x, y, w, h) in faces]
            largest_idx = np.argmax(areas)
            x, y, w, h = faces[largest_idx]
            
            # Expand bbox slightly
            x, y, w, h = expand_bbox((x, y, w, h), frame.shape)
            
            # Extract face region
            face = frame[y:y+h, x:x+w]
            
            if face.size > 0:
                # Run inference
                result = detector.predict(face)
                
                # Store the prediction
                prediction_results.append({
                    'label': result['label'],
                    'confidence': result['confidence'],
                    'timestamp': current_time - start_time
                })
                capture_count += 1
                
                # Count current results
                smile_count = sum(1 for p in prediction_results if p['label'] == 'smile')
                frown_count = sum(1 for p in prediction_results if p['label'] == 'frown')
                
                # Console output
                print(f'{capture_count:<10} | {result["label"].upper():<12} | {result["confidence"]:<10.2f} | Smile: {smile_count}, Frown: {frown_count}')
                
                current_prediction = result['label']
                last_capture_time = current_time
        
        # Visualize current frame
        display_frame = frame.copy()
        if len(faces) > 0:
            x, y, w, h = faces[0]
            x, y, w, h = expand_bbox((x, y, w, h), frame.shape)
            
            # Color based on last capture if available
            if current_prediction == 'smile':
                color = (0, 255, 0)  # Green
            elif current_prediction == 'frown':
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 255)  # Yellow (waiting)
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
        
        # Calculate current counts
        smile_count = sum(1 for p in prediction_results if p['label'] == 'smile')
        frown_count = sum(1 for p in prediction_results if p['label'] == 'frown')
        total_samples = len(prediction_results)
        
        # Time until next capture
        time_to_next = max(0, capture_interval - (current_time - last_capture_time))
        
        # Display stats on frame
        elapsed_time = current_time - start_time
        y_offset = 30
        cv2.putText(display_frame, f'Samples: {capture_count} | Elapsed: {elapsed_time:.1f}s | Next: {time_to_next:.1f}s', 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if total_samples > 0:
            y_offset += 30
            cv2.putText(display_frame, f'SMILE: {smile_count} ({smile_count/total_samples*100:.1f}%)', 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(display_frame, f'FROWN: {frown_count} ({frown_count/total_samples*100:.1f}%)', 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        y_offset += 30
        cv2.putText(display_frame, 'Press SPACEBAR or Q to finish', 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Real-Time Smile/Frown Detector', display_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q') or key == 27:  # Space, Q, or ESC
            break
    
    # Cleanup
    print('\n\nAssessment Complete')
    cap.release()
    cv2.destroyAllWindows()
    
    # Analyze results
    if len(prediction_results) == 0:
        print('No samples captured. Cannot make determination.')
        return 'Insufficient Data'
    
    # Count smiles and frowns
    smile_count = sum(1 for p in prediction_results if p['label'] == 'smile')
    frown_count = sum(1 for p in prediction_results if p['label'] == 'frown')
    total_samples = len(prediction_results)
    
    print(f'\nTotal samples: {total_samples}')
    print(f'Smiles: {smile_count} ({smile_count/total_samples*100:.1f}%)')
    print(f'Frowns: {frown_count} ({frown_count/total_samples*100:.1f}%)')
    
    # Determine final result
    if smile_count > frown_count:
        result = 'Approve'
        print('\nRESULT: APPROVE')
    elif frown_count > smile_count:
        result = 'Disapprove'
        print('\nRESULT: DISAPPROVE')
    else:
        result = 'Tie'
        print('\nRESULT: TIE (Equal smiles and frowns)')
    
    print(f'\nReturning: {result}')
    return result

if __name__ == '__main__':
    result = realtime_smile_detector()
    print(f'\nFinal result: {result}')