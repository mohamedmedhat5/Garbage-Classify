import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = "processed"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return transforms.functional.pad(image, padding, 0, 'constant')

test_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model(num_classes):
    model = models.mobilenet_v2(weights=None) 
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model

def evaluate_model():
    print(f"Starting Evaluation on Device: {DEVICE}")
    
    test_dir = os.path.join(DATA_DIR, 'test')
    if not os.path.exists(test_dir):
        print("Error: 'test' folder not found!")
        return

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_names = test_dataset.classes
    
    print(f"Test Images: {len(test_dataset)}")
    
    try:
        model = get_model(len(class_names))
        model.load_state_dict(torch.load("model.pth", map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    all_preds = []
    all_labels = []

    print("Running Inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("\n" + "="*50)
    print("Final Classification Report")
    print("="*50)
    
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    preds_tensor = torch.tensor(all_preds)
    labels_tensor = torch.tensor(all_labels)
    accuracy = torch.sum(preds_tensor == labels_tensor).item() / len(all_labels)
    
    print(f"\nOverall Test Accuracy: {accuracy:.2%}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_model()