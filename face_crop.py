import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

real_input = "dataset/train/real"
fake_input = "dataset/train/fake"

real_output = "dataset/train_faces/real"
fake_output = "dataset/train_faces/fake"

os.makedirs(real_output, exist_ok=True)
os.makedirs(fake_output, exist_ok=True)

def crop_faces(input_folder, output_folder):
    for img_name in os.listdir(input_folder):

        if not img_name.endswith(".jpg"):
            continue

        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path).convert("RGB")

        # 🔥 THIS LINE WAS MISSING
        face = mtcnn(img)

        if face is not None:
            face_img = face.permute(1, 2, 0).numpy() * 255
            face_img = Image.fromarray(face_img.astype("uint8"))
            face_img.save(os.path.join(output_folder, img_name))

print("Cropping REAL faces...")
crop_faces(real_input, real_output)

print("Cropping FAKE faces...")
crop_faces(fake_input, fake_output)

print("Done!")
