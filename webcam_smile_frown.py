import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import os

# ----------------------------
# Load model
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load("smile_frown_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# Load class mapping
# ----------------------------
if os.path.exists("class_mapping.json"):
    with open("class_mapping.json", "r") as f:
        class_to_idx = json.load(f)
    # Create reverse mapping: index -> class name
    labels = [None] * len(class_to_idx)
    for class_name, idx in class_to_idx.items():
        labels[idx] = class_name
    print(f"Loaded class mapping: {class_to_idx}")
    print(f"Labels array: {labels}")
else:
    # Fallback to hardcoded (alphabetical order: frown=0, smile=1)
    labels = ["frown", "smile"]
    print("Warning: class_mapping.json not found. Using default: ['frown', 'smile']")

# ----------------------------
# Preprocessing
# ----------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------
# Webcam Loop
# ----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Convert to PIL for torchvision
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(img_tensor)
            probs = torch.softmax(preds, dim=1)[0]
            _, idx = torch.max(preds, 1)
            label = labels[idx.item()]
            conf = probs[idx].item()
            
            # Debug: print raw predictions (only first time or if stuck)
            # Uncomment to debug:
            # print(f"Raw logits: {preds[0].cpu().numpy()}, Probs: {probs.cpu().numpy()}, Predicted: {idx.item()} ({label})")

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        # Show both probabilities for debugging
        prob_text = " ".join([f"{labels[i]}:{probs[i].item():.2f}" for i in range(len(labels))])
        cv2.putText(frame, f"{label} ({conf:.2f})",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)
        cv2.putText(frame, prob_text,
                    (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,0), 1)

    cv2.imshow("Smile/Frown Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()