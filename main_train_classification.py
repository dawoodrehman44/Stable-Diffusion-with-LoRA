import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from image_classification import CXRClassification
from classification_dataset import CheXpertDataset
from evaluation_metrics import evaluate_all_metrics
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Config
epochs = 100
patience = 10
min_epochs = 50
weight_decay = 1e-4
initial_lr = 1e-4

# Data
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dataset = CheXpertDataset(split='train', transform=transform)
val_dataset = CheXpertDataset(split='val', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model
model_config = {
    "load_backbone_weights": "checkpoints/cxrclip_mc/r50_mc.pt", # can be changed according to the model, and experiments
    "freeze_backbone_weights": False,
    "image_encoder": {
        "name": "resnet",
        "resnet_type": "resnet50",
        "pretrained": True,
        "source": "cxr_clip"
    },
    "classifier": {
        "config": {
            "name": "linear",
            "n_class": 14
        }
    }
}
model = CXRClassification(model_config=model_config, model_type="resnet").cuda()

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Early Stopping
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model({"images": images, "labels": labels})["cls_pred"]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images, labels = images.cuda(), labels.cuda()
            outputs = model({"images": images, "labels": labels})["cls_pred"]
            val_loss += criterion(outputs, labels).item()
    val_loss /= len(val_loader)
    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "classification_best_model.pth")
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience and epoch >= min_epochs:
        print("Early stopping triggered.")
        break

# Evaluation
model.load_state_dict(torch.load("classification_best_model.pth"))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for batch in val_loader:
        images, labels = batch
        images = images.cuda()
        outputs = model({"images": images, "labels": labels.cuda()})["cls_pred"].sigmoid().cpu().numpy()
        y_true.append(labels.numpy())
        y_pred.append(outputs)

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

evaluate_all_metrics(y_true, y_pred)
