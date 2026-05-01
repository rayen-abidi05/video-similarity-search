import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import numpy as np
from google.colab import drive
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETUP & MODELS ---
drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()

# Individual transform steps (as you had them originally)
resize = transforms.Resize(256)
crop = transforms.CenterCrop(224)
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def my_preprocess_function(img):
    """Your original transform logic"""
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)
    return img

def get_embedding_from_frame(cv2_frame):
    """Converts a cv2 frame to an embedding vector using your preprocess function."""
    img_rgb = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Using your specific function here
    img_tensor = my_preprocess_function(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = feature_extractor(img_tensor)
    return embedding.cpu().flatten().numpy()

# --- 3. MAIN PROCESSING LOOP ---

video_folder = '/content/drive/MyDrive/Videos/Nature'
video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.avi', '.mp4'))]

if not video_files:
    print("No videos found!")
else:
    SIMILARITY_THRESHOLD = 0.98

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]

        # We use a fresh dictionary for every video to save them as separate .npy files
        current_video_data = {}

        print(f"\nProcessing: {video_file}")
        cap = cv2.VideoCapture(video_path)

        unique_vectors = []
        frame_count = 0
        accepted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Step 1: Uniform Sampling (Every 10th frame)
            if frame_count % 10 == 0:
                new_vector = get_embedding_from_frame(frame)

                # Step 2: Redundancy Filtering (The "Keyframe" logic)
                if len(unique_vectors) == 0:
                    unique_vectors.append(new_vector)
                    accepted_count += 1
                else:
                    last_vector = unique_vectors[-1]
                    sim = cosine_similarity(new_vector.reshape(1, -1), last_vector.reshape(1, -1))[0][0]

                    if sim < SIMILARITY_THRESHOLD:
                        unique_vectors.append(new_vector)
                        accepted_count += 1
            frame_count += 1

        cap.release()

        # Step 3: Mean Pooling & Saving
        if unique_vectors:
            mean_vector = np.mean(unique_vectors, axis=0)

            # Save this video's vector into a dictionary
            current_video_data[video_name] = mean_vector

            # --- 4. SAVE RESULTS ---
            save_dir = os.path.join(video_folder, "data")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = os.path.join(save_dir, f'{video_name}.npy')
            np.save(save_path, current_video_data)

            print(f"Captured {accepted_count} unique keyframes.")
            print(f"Saved vector to: {save_path}")

print("\nAll videos processed successfully.")