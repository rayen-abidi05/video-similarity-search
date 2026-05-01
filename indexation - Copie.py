import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import numpy as np
from google.colab import drive

# --- 1. SETUP & MODELS ---
drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()

# Individual transform components
resize = transforms.Resize(256)
crop = transforms.CenterCrop(224)
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --- 2. CORE FUNCTIONS ---

def my_preprocess_function(img):
    """The function version of transforms"""
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)
    return img

def get_embedding_from_frame(cv2_frame):
    """Processes a single BGR frame using the preprocess function."""
    img_rgb = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Using it as a function call here
    img_tensor = my_preprocess_function(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = feature_extractor(img_tensor)
    return embedding.cpu().flatten().numpy()

def get_video_query_vector(video_path):
    """Loops through video frames and returns the MEAN vector."""
    cap = cv2.VideoCapture(video_path)
    vectors = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_count % 10 == 0: 
            vectors.append(get_embedding_from_frame(frame))
        frame_count += 1
    cap.release()
    return np.mean(vectors, axis=0) if vectors else None

def calculate_cosine_similarity(vector_A, vector_B):
    dot_product = np.dot(vector_A, vector_B)
    norm_A = np.linalg.norm(vector_A)
    norm_B = np.linalg.norm(vector_B)
    return dot_product / (norm_A * norm_B + 1e-9)

# --- 3. LOADING THE DATABASE ---

vector_folder = '/content/drive/MyDrive/Videos/Nature/data'
all_video_features = {}

if os.path.exists(vector_folder):
    vector_files = [f for f in os.listdir(vector_folder) if f.endswith('.npy')]
    for f in vector_files:
        video_name = f.replace('.npy', '')
        file_path = os.path.join(vector_folder, f)
        
        # Load and handle the dictionary-in-npy issue
        data = np.load(file_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == 'object':
            data_dict = data.item() 
            vector = list(data_dict.values())[0]
        else:
            vector = data
            
        all_video_features[video_name] = vector
    print(f"Loaded {len(all_video_features)} database vectors.")

# --- 4. RUNNING THE SEARCH ON TEST FOLDER ---

# Updated path to the 'test' folder
test_folder = '/content/drive/MyDrive/Videos/test'

if os.path.exists(test_folder):
    test_videos = [f for f in os.listdir(test_folder) if f.lower().endswith(('.mp4', '.avi'))]
    
    for test_video in test_videos:
        input_video_path = os.path.join(test_folder, test_video)
        print(f"\n--- TESTING VIDEO: {test_video} ---")
        
        query_vector = get_video_query_vector(input_video_path) 

        if query_vector is not None:
            results = []
            for name, db_vector in all_video_features.items():
                score = calculate_cosine_similarity(query_vector, db_vector)
                results.append((name, score))

            # Sort by highest score
            results.sort(key=lambda x: x[1], reverse=True)

            print(f"Top results for {test_video}:")
            for i, (name, score) in enumerate(results[:3]):
                print(f"  {i+1}. {name} | Score: {score:.4f}")
else:
    print(f"Test folder not found at {test_folder}")