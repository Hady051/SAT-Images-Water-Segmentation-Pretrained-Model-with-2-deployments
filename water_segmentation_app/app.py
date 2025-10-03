# app.py
import os
import shutil
import base64
from flask import Flask, render_template, request, redirect, url_for, flash
import torch
import numpy as np
import rasterio
import cv2
import segmentation_models_pytorch as smp

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flash messages
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

## Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

## Model paths (adjust these to your local paths)
best_model_path_val_iou = 'best_model_val_iou_version_8.pth'  # Path to your model
mins_path = 'mins.npy'  # Path to mins.npy
maxs_path = 'maxs.npy'  # Path to maxs.npy

## Load mins and maxs
try:
    mins = np.load(mins_path)
    maxs = np.load(maxs_path)
    print("Mins and maxs loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure mins.npy and maxs.npy are in the project directory.")
    exit(1)

## Model configuration
model_in_channels = 17  # Assuming 12 bands + 5 indices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Define the model architecture
def get_model_resnet34(num_channels, classes=1):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=num_channels,
        classes=classes,
        activation=None
    )
    # Fix conv1 mapping for RGB channels
    conv1 = model.encoder.conv1
    red_idx, green_idx, blue_idx = 3, 2, 1
    with torch.no_grad():
        conv1.weight[:, red_idx] = conv1.weight[:, 0].clone()
        conv1.weight[:, green_idx] = conv1.weight[:, 1].clone()
        conv1.weight[:, blue_idx] = conv1.weight[:, 2].clone()
        torch.nn.init.kaiming_normal_(conv1.weight[:, 0:3])
    return model

## Load the model
try:
    model = get_model_resnet34(num_channels=model_in_channels)
    model.load_state_dict(torch.load(best_model_path_val_iou, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure the model file exists.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

## Function to load multispectral image
def load_image_for_inference(path):
    with rasterio.open(path) as src:
        img = src.read()  # (bands, height, width)
        img = img.astype(np.float32)
    return img

## Function to normalize image
def normalize_image_for_inference(img, mins, maxs):
    if img.shape[0] != len(mins):
        raise ValueError(f"Image has {img.shape[0]} bands, but mins/maxs have {len(mins)}.")
    normalized_img = np.copy(img)
    for c in range(len(mins)):
        denominator = (maxs[c] - mins[c])
        if denominator == 0:
            normalized_img[c] = 0
        else:
            normalized_img[c] = (img[c] - mins[c]) / (denominator + 1e-6)
    return normalized_img

# Function to compute water indices
def compute_water_indices_for_inference(img):
    blue = img[1]
    green = img[2]
    red = img[3]
    nir = img[4]
    swir1 = img[5]
    swir2 = img[6]

    ndwi = (green - nir) / (green + nir + 1e-10)
    mndwi = (green - swir1) / (green + swir1 + 1e-10)
    awei_sh = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
    awei_ns = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2
    ndvi = (nir - red) / (nir + red + 1e-10)

    indices = np.stack([ndwi, mndwi, awei_sh, awei_ns, ndvi], axis=0)
    img_with_indices = np.concatenate([img, indices], axis=0)
    return img_with_indices

## Function to perform inference
def inference_on_tif(tif_path, mins, maxs, device, img_size=128, include_indices=True):
    img = load_image_for_inference(tif_path)
    if img.shape[1] != img_size or img.shape[2] != img_size:
        print(f"Resizing image from {img.shape[1]}x{img.shape[2]} to {img_size}x{img_size}.")
        resized_img = np.zeros((img.shape[0], img_size, img_size), dtype=img.dtype)
        for c in range(img.shape[0]):
            resized_img[c] = cv2.resize(img[c], (img_size, img_size), interpolation=cv2.INTER_AREA)
        img = resized_img
    img = normalize_image_for_inference(img, mins, maxs)
    if include_indices:
        img = compute_water_indices_for_inference(img)
    if img.shape[0] != model_in_channels:
        raise ValueError(f"Processed image has {img.shape[0]} channels, but model expects {model_in_channels}.")
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    probs = torch.sigmoid(output)
    pred_mask = (probs.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return pred_mask, load_image_for_inference(tif_path)  # Return mask and original for RGB

## Function to generate RGB approximation
def generate_rgb_image(original_img):
    if original_img.shape[0] > 3:
        rgb_bands = original_img[[3, 2, 1]]  # Red, Green, Blue
        rgb_bands = (rgb_bands - rgb_bands.min()) / (rgb_bands.max() - rgb_bands.min() + 1e-6)
        rgb_image = np.transpose(rgb_bands, (1, 2, 0)) * 255
        return rgb_image.astype(np.uint8)
    return None

## Function to generate colored mask
def generate_colored_mask(pred_mask):
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    colored_mask[pred_mask == 1] = [0, 0, 255]  # Blue for water
    return colored_mask

## Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    rgb_base64 = None
    mask_base64 = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and file.filename.lower().endswith(('.tif', '.tiff')):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                pred_mask, original_img = inference_on_tif(filepath, mins, maxs, device)
                rgb_image = generate_rgb_image(original_img)
                colored_mask = generate_colored_mask(pred_mask)

                # 🔹 Resize for display (does NOT affect model)
                display_size = (256, 256)  # Change from 128 to 256, 512, etc. as you like
                if rgb_image is not None:
                    rgb_image = cv2.resize(rgb_image, display_size, interpolation=cv2.INTER_NEAREST)
                colored_mask = cv2.resize(colored_mask, display_size, interpolation=cv2.INTER_NEAREST)

                # Save images temporarily
                rgb_path = os.path.join(app.config['RESULTS_FOLDER'], 'rgb.png')
                mask_path = os.path.join(app.config['RESULTS_FOLDER'], 'mask.png')
                if rgb_image is not None:
                    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                    with open(rgb_path, 'rb') as f:
                        rgb_base64 = base64.b64encode(f.read()).decode('utf-8')
                cv2.imwrite(mask_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                with open(mask_path, 'rb') as f:
                    mask_base64 = base64.b64encode(f.read()).decode('utf-8')
                # Clean up
                os.remove(filepath)
                os.remove(rgb_path)
                os.remove(mask_path)
            except Exception as e:
                flash(f'Error processing file: {e}')
                os.remove(filepath)
    return render_template('index.html', rgb_base64=rgb_base64, mask_base64=mask_base64)

if __name__ == '__main__':
    app.run(debug=True)