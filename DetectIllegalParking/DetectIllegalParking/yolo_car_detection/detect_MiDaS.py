import torch
import cv2
import numpy as np
from transformers import pipeline
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load MiDaS model
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def estimate_depth(image):

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Apply the necessary transformations
    input_tensor = transform(pil_image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Forward pass through MiDaS model
    with torch.no_grad():
        depth_map = model(input_tensor)

    # Normalize depth map
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize to [0, 1]
    return depth_map


###############################


#image_path = "C:/Users/Filippo/Documents/DSDM/Computer Vision/Car_3D_detection/images/leftImg8bit/val/frankfurt/frankfurt_000000_005543_leftImg8bit.png"
#image = cv2.imread(image_path)  # Load image in BGR format

# Estimate the depth map
#depth_map = estimate_depth(image)

# Visualize the depth map
#plt.imshow(depth_map, cmap='inferno')
#plt.colorbar()
#plt.title('Depth Map')
#plt.show()

