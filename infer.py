import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import json

# Argument parser
parser = argparse.ArgumentParser(description="Inference with ResNet50 on a single image.")
parser.add_argument('image_path', type=str, help='Path to the image (JPG)')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
args = parser.parse_args()

# Load class mapping
with open('classmap.json', 'r') as f:
    idx_to_class = json.load(f)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.efficientnet_b3(pretrained=False)

# Replace final FC layer
num_classes = len(idx_to_class)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

state_dict = torch.load("checkpoints/best_model.pth", map_location=device)
# Strip 'module.' prefix if present (from DataParallel)
if any(k.startswith('module.') for k in state_dict['model_state_dict'].keys()):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict['model_state_dict'].items():
        new_key = k.replace('module.', '', 1)  # remove only first 'module.'
        new_state_dict[new_key] = v
    state_dict = new_state_dict
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Preprocessing pipeline (should match training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Load and preprocess image
if not os.path.exists(args.image_path):
    raise FileNotFoundError(f"Image not found: {args.image_path}")

image = Image.open(args.image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    # get top 5 predictions
    top5_probs, top5_indices = torch.topk(probs, 5)
    top5_classes = [idx_to_class[str(idx.item())] for idx in top5_indices]
    top5_probs = top5_probs.cpu().numpy()

# Output result
print("Top 5 predictions:")
for i in range(5):
    print(f"{i+1}: {top5_classes[i]} - {top5_probs[i]*100:.2f}%")