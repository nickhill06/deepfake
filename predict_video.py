import torch
import cv2
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification
from facenet_pytorch import MTCNN
from torchvision import transforms
import os

# --- CONFIGURATION ---
MODEL_PATH = "vit_model_balanced.pth"
# REPLACE WITH THE PATH TO YOUR FAKE VIDEO
VIDEO_PATH = "WhatsApp Video 2026-02-17 at 1.54.08 PM.mp4"
# ---------------------

def main():
    print("--- STARTING PREDICTION ---")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # Load the weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Setup MTCNN (Face Detector)
    # We use default settings because that's what face_crop.py used
    mtcnn = MTCNN(keep_all=False, device=device)

    # 4. Setup Transforms (Must match training loader)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 5. Process Video
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    fake_votes = 0
    real_votes = 0
    frame_count = 0
    detected_count = 0

    print(f"Processing video: {os.path.basename(VIDEO_PATH)}")
    print("Reading frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check every 5th frame to speed it up
        if frame_count % 5 != 0:
            frame_count += 1
            continue

        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # --- CRITICAL STEP: REPLICATE TRAINING DATA PROCESSING ---
        # 1. Get the tensor from MTCNN (This is normalized [-1, 1])
        face_tensor = mtcnn(pil_img)

        if face_tensor is not None:
            detected_count += 1
            
            # 2. Replicate the exact math from face_crop.py
            # permute moves channels to end: (3, 224, 224) -> (224, 224, 3)
            # * 255 scales it up
            # .astype('uint8') casts it (this creates the 'corruption' effect)
            face_numpy = face_tensor.permute(1, 2, 0).cpu().numpy() * 255
            face_uint8 = face_numpy.astype('uint8')
            
            # 3. Convert back to Image for the Transform pipeline
            face_pil = Image.fromarray(face_uint8)

            # 4. Prepare for Model
            input_tensor = transform(face_pil).unsqueeze(0).to(device)

            # 5. Predict
            with torch.no_grad():
                outputs = model(input_tensor).logits
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get the probability of it being Fake (Class 1)
                fake_prob = probs[0][1].item()
                
                # STRICTER THRESHOLD:
                # If the model is even 50% sure it's fake, count it.
                if fake_prob > 0.5: 
                    prediction = 1 # Fake
                else:
                    prediction = 0 # Real
                
                if prediction == 1:
                    fake_votes += 1
                    # print(f"Frame {frame_count}: 🛑 FAKE ({fake_prob:.2f})")
                else:
                    real_votes += 1
                    # print(f"Frame {frame_count}: ✅ REAL ({1-fake_prob:.2f})")

        frame_count += 1

    cap.release()

    # 6. Final Results
    total_votes = real_votes + fake_votes
    print("\n" + "="*30)
    print(f"Frames Analyzed: {detected_count}")
    print(f"Real Votes:      {real_votes}")
    print(f"Fake Votes:      {fake_votes}")
    print("="*30)

    if total_votes == 0:
        print("⚠️ No faces detected.")
    elif fake_votes>real_votes: 
        # If we see even a few fake frames, flag the video
        # (Deepfakes often flicker, so finding *any* fake frames is suspicious)
        print(f">>> RESULT: 🛑 FAKE VIDEO 🛑 (Found {fake_votes} suspicious frames)")
    else:
        print(">>> RESULT: ✅ REAL VIDEO ✅")

if __name__ == "__main__":
    main()