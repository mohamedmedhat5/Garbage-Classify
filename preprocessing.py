import os
import shutil
import cv2
import imagehash
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RAW_DATA_DIR = "raw"
OUTPUT_DIR = "processed"
BLUR_THRESHOLD = 100.0
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def is_corrupt(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return False
    except:
        return True

def is_blurry(filepath):
    try:
        image = cv2.imread(filepath)
        if image is None: return True
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < BLUR_THRESHOLD
    except:
        return True

def get_image_hash(filepath):
    try:
        with Image.open(filepath) as img:
            return str(imagehash.phash(img))
    except:
        return None

def process_dataset():
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning old output directory: {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data folder '{RAW_DATA_DIR}' not found!")
        return

    classes = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    print(f"Found Classes: {classes}")

    global_hashes = set()

    for class_name in classes:
        print(f"\nProcessing Class: {class_name}...")
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        
        all_files = [f for f in os.listdir(class_dir) if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]
        
        valid_images = []
        
        for fname in tqdm(all_files, desc=f"Filtering {class_name}"):
            fpath = os.path.join(class_dir, fname)
            
            if is_corrupt(fpath):
                continue
            
            if is_blurry(fpath):
                continue

            img_hash = get_image_hash(fpath)
            if img_hash is None or img_hash in global_hashes:
                continue
            
            global_hashes.add(img_hash)
            valid_images.append(fname)

        count = len(valid_images)
        if count < 5:
            print(f"Warning: Class {class_name} has only {count} valid images. Skipping split.")
            continue

        train_imgs, temp_imgs = train_test_split(
            valid_images, test_size=(1 - TRAIN_RATIO), random_state=42, shuffle=True
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=0.5, random_state=42, shuffle=True
        )

        print(f"Stats: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

        def save_files(file_list, split_name):
            save_path = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(save_path, exist_ok=True)
            
            for idx, filename in enumerate(file_list):
                src = os.path.join(class_dir, filename)
                dst = os.path.join(save_path, f"{class_name}_{idx:04d}.jpg")
                
                try:
                    with Image.open(src) as img:
                        img.convert('RGB').save(dst, 'JPEG', quality=95)
                except Exception as e:
                    print(f"Error saving {filename}: {e}")

        save_files(train_imgs, 'train')
        save_files(val_imgs, 'val')
        save_files(test_imgs, 'test')

    print("\nPreprocessing Pipeline Completed Successfully!")
    print(f"Clean Data is ready at: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    process_dataset()