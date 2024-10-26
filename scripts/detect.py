import os
import cv2
from pathlib import Path
import torch

# Load YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'yolov7')

# Directory with images
image_folder = Path(r"C:\Users\fazba\Downloads\Preannotated Images_v2")
output_folder = Path("output")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for img_path in image_folder.glob("*.jpg"):  # or "*.png" depending on your images
    img = cv2.imread(str(img_path))
    results = model(img)

    # Save the resulting image with bounding boxes
    results.render()  # Draw bounding boxes
    output_image_path = output_folder / img_path.name
    cv2.imwrite(str(output_image_path), results.imgs[0])
