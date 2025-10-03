# Satellite Images Water Segmentation

This project implements a **water body segmentation system** using a **fine-tuned ResNet-34 U-Net model** on multispectral (12 channels) satellite imagery.
The ground truth Masks that the model was trained on show **water that isn't visible on the satellite images to the human eye** like underground water as well as **visible water bodies**. 
The model predicts water masks from multispectral `.tif` images by leveraging both spectral bands and water indices.  

---

## 📊 Model Performance (Evaluation Metrics)

The fine-tuned **ResNet-34** model achieved the following results on the validation set:

- **IoU (Intersection over Union):** `0.8589`  
- **Precision:** `0.9350`  
- **Recall:** `0.9105`  
- **F1 Score:** `0.9226`  

These metrics show that the model provides **robust segmentation quality**, detecting water bodies accurately with high precision and recall.

---

⚠️ **Note:** The trained model weights file (`best_model_val_iou_version_8.pth`) is **not included** here due to size limitations (~95 MB). To run inference, you must place this file in the project root.

## Deployment Methods

This project includes **two deployment approaches**:

### 1. Flask + HTML/CSS (Web App)
- Backend built with **Flask** (`app.py`).
- Frontend includes:
  - `index.html` for layout and image upload.
  - `style.css` for styling, with a **golden-brown theme** for custom buttons.
- Workflow:
  1. Upload a multispectral `.tif` image.
  2. The system generates an **RGB approximation**.
  3. The model outputs a **predicted water segmentation mask**.
- Both images are displayed side-by-side in the browser.

---

### 2. Streamlit + Ngrok (Notebook-based Deployment in **Colab**)
An alternative deployment method is provided directly inside the Jupyter Notebook (no Flask/HTML/CSS required).  

- Uses **Streamlit** for the frontend (`app.py` cell in the notebook).  
- Integrates **Ngrok** for public URL tunneling.  
- Features:
  - File uploader (`st.file_uploader`) for `.tif` images.
  - Displays **RGB approximation** and **predicted water mask**.
  - Side-by-side comparison of original and mask.
  - Optional **Ngrok tunnel** to share app externally.


---

## Pictures of the Flask, Html and CSS Deployment

<img width="1366" height="690" alt="Screenshot (1635)" src="https://github.com/user-attachments/assets/5d1da470-a605-4f1b-877e-656577c0dd7f" />

<img width="1366" height="693" alt="Screenshot (1636)" src="https://github.com/user-attachments/assets/b98b2a9f-4105-4d45-9447-2b2cfa0e7ea6" />

<img width="1366" height="693" alt="Screenshot (1637)" src="https://github.com/user-attachments/assets/36fcf74f-9650-44fd-9c3f-2da7fa0c653a" />

<img width="1366" height="691" alt="Screenshot (1638)" src="https://github.com/user-attachments/assets/23e0a236-f5e9-43b6-afcb-e366cce6ab65" />

