import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
from app.utils import get_transform

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.to(next(self.model.parameters()).device)

        # Forward pass
        x.requires_grad_()
        output = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).cpu().detach()
        cam = torch.clamp(cam, min=0)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)

        return cam.numpy()[0]

def visualize_single_image(model, image_path, class_names, device):
    model.eval()

    # Load and transform image
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0, pred.item()].item()

    # Generate Grad-CAM
    grad_cam = GradCAM(model, model.conv3)
    cam = grad_cam(input_tensor, pred.item())

    # Resize CAM
    cam_resized = np.uint8(255 * cam)
    cam_resized = np.repeat(cam_resized[:, :, np.newaxis], 3, axis=2)
    cam_resized = cv2.resize(cam_resized, (150, 150))

    # Create heatmap
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # De-normalize image
    img = input_tensor.cpu().numpy()[0].transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.uint8(255 * img)

    # Create overlay
    alpha = 0.4
    overlay = heatmap * alpha + img * (1 - alpha)
    overlay = overlay / 255.0

    # Plot
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.title(f'Original\nPrediction: {predicted_class}\nConfidence: {confidence:.2%}')
    plt.imshow(img / 255.0)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Grad-CAM Heatmap')
    plt.imshow(heatmap / 255.0)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(overlay)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_folder(model, folder_path, class_names, device, num_images=5):
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]

    if not image_files:
        print(f"No image files found in {folder_path}")
        return

    if len(image_files) > num_images:
        image_files = random.sample(image_files, num_images)

    for image_path in image_files:
        print(f"Visualizing: {os.path.basename(image_path)}")
        visualize_single_image(model, image_path, class_names, device)