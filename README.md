# Deepfake Detection using Vision Transformer (ViT)

## 📌 Project Overview
This project detects whether a video is REAL or FAKE (deepfake) using Deep Learning.  
It extracts frames from videos, detects faces, and uses a Vision Transformer model to classify images.

---

## 🚀 Features
- Video frame extraction using OpenCV
- Face detection using MTCNN
- Deepfake classification using Vision Transformer (ViT)
- CNN (ResNet18) baseline model
- Video-level prediction (Real / Fake output)

---

## 🧠 Tech Stack

### Programming Language
- Python

### Libraries
- PyTorch
- Torchvision
- OpenCV
- HuggingFace Transformers
- facenet-pytorch (MTCNN)
- Pillow (PIL)

---

STRUCTURE 

Deepfake_ViT_Project/
│
├ dataset/
├ dataset/train_faces/
├ models/
│ ├ vit_model.pth
│ ├ cnn_model.pth
│
├ extract_frames.py
├ face_crop.py
├ dataset_loader.py
├ train_vit.py
├ train_cnn.py
├ predict_video.py
└ README.md




---

## 📊 Model Used
- Vision Transformer (google/vit-base-patch16-224)
- ResNet18 (Baseline CNN)

---

## 📌 Future Improvements
- Real-time webcam detection
- Web interface using Streamlit
- Model ensemble (ViT + CNN)
- Accuracy and confusion matrix visualization

---

## 👨‍💻 Author
Your Name


## 📂 Project Structure
