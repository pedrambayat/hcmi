import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from collections import Counter

# data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# loading datasets
train_dir = "dataset/dataset_forrest/train"
test_dir = "dataset/dataset_forrest/test"

train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
test_ds = datasets.ImageFolder(test_dir, transform=test_transform)

# verify class mapping
print(f"Class names: {train_ds.classes}")
print(f"Class to index mapping: {train_ds.class_to_idx}")
print(f"Number of classes: {len(train_ds.classes)}")

# count samples per class
train_labels = [label for _, label in train_ds]
train_class_counts = Counter(train_labels)
print(f"Training samples per class: {dict(zip(train_ds.classes, [train_class_counts[i] for i in range(len(train_ds.classes))]))}")

test_labels = [label for _, label in test_ds]
test_class_counts = Counter(test_labels)
print(f"Test samples per class: {dict(zip(test_ds.classes, [test_class_counts[i] for i in range(len(test_ds.classes))]))}")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

# model (ResNet18)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)   # smile / frown
model = model.to(device)

# training setup
# focal Loss for handling hard examples (focuses on misclassified samples)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# calculate class weights to handle any imbalance (inverse frequency)
total_samples = sum(train_class_counts.values())
class_weights = torch.tensor([total_samples / (len(train_ds.classes) * train_class_counts[i]) 
                              for i in range(len(train_ds.classes))], dtype=torch.float32).to(device)
print(f"Class weights: {dict(zip(train_ds.classes, class_weights.cpu().numpy()))}")

# use Focal Loss to focus on hard examples (helps with frown detection)
# gamma=2.0 focuses more on hard examples, alpha can be adjusted per class if needed
criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
num_epochs = 4

# train loop with validation monitoring
best_test_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # quick validation check every few epochs with per-class metrics
    if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
        model.eval()
        val_correct, val_total = 0, 0
        val_class_correct = [0] * len(test_ds.classes)
        val_class_total = [0] * len(test_ds.classes)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                _, predicted = torch.max(preds, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                # per-class validation
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    val_class_total[label] += 1
                    if predicted[i] == labels[i]:
                        val_class_correct[label] += 1
        val_acc = val_correct / val_total * 100
        scheduler.step(val_acc)  # adjust learning rate based on validation accuracy
        model.train()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Per-class Val: {', '.join([f'{test_ds.classes[i]}:{100*val_class_correct[i]/val_class_total[i]:.1f}%' if val_class_total[i] > 0 else f'{test_ds.classes[i]}:N/A' for i in range(len(test_ds.classes))])}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# test accuracy with per-class metrics
model.eval()
correct, total = 0, 0
class_correct = [0] * len(test_ds.classes)
class_total = [0] * len(test_ds.classes)

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        _, predicted = torch.max(preds, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # per-class accuracy
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == labels[i]:
                class_correct[label] += 1

accuracy = correct / total * 100
print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
print("Per-class Test Accuracy:")
for i, class_name in enumerate(test_ds.classes):
    if class_total[i] > 0:
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f"  {class_name}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"  {class_name}: No samples")

# save model
torch.save(model.state_dict(), "smile_frown_resnet18.pth")
print("Model saved as smile_frown_resnet18.pth")

# save class mapping for reference
import json
with open("class_mapping.json", "w") as f:
    json.dump(train_ds.class_to_idx, f)
print(f"Class mapping saved: {train_ds.class_to_idx}")