# Smart-Eye: Intelligent Waste Classification System

## Project Overview
Smart-Eye is a Deep Learning-based computer vision system designed to classify waste materials into 12 distinct categories to assist in automated recycling processes. The system leverages Transfer Learning using the MobileNetV2 architecture, achieving a test accuracy of 95.6%. It includes a full-stack implementation with a FastAPI backend for real-time inference and a web-based dashboard for visualization.

## Key Features
- **Architecture:** MobileNetV2 (Pre-trained on ImageNet) with a custom classification head.
- **Accuracy:** 95.6% on the unseen test dataset.
- **Classes:** 12 categories (battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass).
- **Data Pipeline:** Automated cleaning pipeline including de-duplication (pHash), blur detection (Laplacian Variance), and aspect-ratio preserving resizing.
- **Handling Imbalance:** Implemented WeightedRandomSampler to ensure balanced training across rare classes.
- **Real-Time Interface:** WebSocket-based communication for low-latency inference.

## Technology Stack
- **Deep Learning:** PyTorch, Torchvision
- **Backend:** FastAPI, Uvicorn, WebSockets
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Data Processing:** OpenCV, PIL, ImageHash, Scikit-learn
- **Visualization:** Matplotlib, Seaborn

## Project Structure
- `raw/`: Directory for initial raw data collection.
- `processed/`: Directory containing cleaned and split datasets (train/val/test).
- `preprocessing.py`: Script for cleaning, filtering, and splitting the raw dataset.
- `training.py`: Script for model training, including data augmentation and saving the best weights.
- `evaluate.py`: Script for generating classification reports and the confusion matrix.
- `api.py`: The FastAPI backend server handling model loading and inference requests.
- `index.html`: The user interface for the dashboard.
- `run_app.bat`: Windows batch script to launch the application.

## Installation

1. Create a virtual environment (optional but recommended):
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies:
   pip install torch torchvision fastapi uvicorn[standard] opencv-python imagehash pillow scikit-learn matplotlib seaborn tqdm

## Usage Guide

### 1. Data Preparation
If you have raw images in the `raw/` folder, run the preprocessing pipeline:
python preprocessing.py

### 2. Training the Model
To retrain the model from scratch:
python training.py

### 3. Evaluation
To generate the performance metrics and confusion matrix:
python evaluate.py

### 4. Running the Application
To start the server and open the dashboard automatically:
run_app.bat

## Team Members

**Rahma Reda**,
**Renad Ayman**,
**Manar Magdy**,
**Osama El-Gohary**,
**Mohamed Alaa** $
**Mohamed Medhat**

