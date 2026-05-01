# Video Similarity Search Engine

A content-based video retrieval system built in Python using a pre-trained ResNet-50 model. Given a query video, the system finds and ranks the most visually similar videos from a dataset using cosine similarity on deep feature embeddings.

---

## How It Works

1. **Keyframe Extraction** — Every 10th frame is sampled from each video
2. **Redundancy Filtering** — Near-duplicate frames are removed using a cosine similarity threshold (0.98)
3. **Feature Extraction** — Each keyframe is encoded into a 2048-dim vector using ResNet-50 (without the final classification layer)
4. **Mean Pooling** — All keyframe vectors are averaged into a single fixed-size vector per video
5. **Indexing** — Each video's vector is saved as a `.npy` file on Google Drive
6. **Search** — A query video goes through the same pipeline and is compared against the indexed database using cosine similarity, returning the top matches

---

## Tech Stack

- **Python** — core language
- **PyTorch** — deep learning framework
- **ResNet-50** — pre-trained CNN for feature extraction (ImageNet weights)
- **OpenCV (cv2)** — video reading and frame extraction
- **NumPy** — vector storage and mean pooling
- **Pillow (PIL)** — image preprocessing
- **scikit-learn** — cosine similarity (indexing phase)
- **Google Colab** — GPU runtime environment
- **Google Drive** — video dataset and vector storage

---

## Project Structure

```
Video-Similarity-Search/
├── cell1_indexing.py      # Extracts and indexes video embeddings → saves .npy files
├── cell2_search.py        # Loads indexed vectors, processes query video, returns top results
└── README.md
```

> Originally developed as a 2-cell Google Colab notebook.

---

## Pipeline Overview

```
Input Video
     │
     ▼
Frame Sampling (every 10th frame)
     │
     ▼
Redundancy Filtering (cosine sim < 0.98)
     │
     ▼
ResNet-50 Feature Extraction (2048-dim vector per frame)
     │
     ▼
Mean Pooling → Single video vector
     │
     ▼
Cosine Similarity vs Database
     │
     ▼
Top N Most Similar Videos
```

---

## Getting Started

### Prerequisites

- Google Colab account (free GPU recommended)
- Google Drive with videos organized as:
```
MyDrive/
└── Videos/
    ├── Nature/          # Dataset videos (.mp4 or .avi)
    │   └── data/        # Auto-created: stores .npy vector files
    └── test/            # Query videos to search with
```

### Installation (Colab)

```python
!pip install torch torchvision opencv-python-headless scikit-learn Pillow numpy
```

### Usage

**Step 1 — Index your dataset** (run `cell1_indexing.py`):
- Processes all videos in `Videos/Nature/`
- Saves one `.npy` vector file per video in `Videos/Nature/data/`

**Step 2 — Search** (run `cell2_search.py`):
- Loads all indexed vectors
- Processes each video in `Videos/test/`
- Prints top 3 most similar videos with similarity scores

---

## Example Output

```
--- TESTING VIDEO: query_clip.mp4 ---
Top results for query_clip.mp4:
  1. forest_walk   | Score: 0.9821
  2. river_stream  | Score: 0.9654
  3. mountain_view | Score: 0.9102
```

---

## Author

**Rayen Abidi**  
[LinkedIn](https://www.linkedin.com/in/rayen-abidi-9610913bb) · [GitHub](https://github.com/Abidi-Rayen)
