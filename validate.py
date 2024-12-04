import os
from PIL import Image
import torch
from torchvision.transforms import transforms
import torch.nn as nn

import matplotlib.pyplot as plt



def calculate_mean_std(image_dir):
    """
    Calculates the mean and standard deviation for a dataset of images.
    Args:
        image_dir (str): Path to the image directory.
    Returns:
        tuple: Mean and std for each channel (R, G, B).
    """
    transform = transforms.ToTensor()  # Convert images to tensors
    means = []
    stds = []

    # List all image paths
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]

    # Accumulate sum and squared sum for mean/std calculation
    pixel_sum = torch.zeros(3)  # For RGB channels
    pixel_squared_sum = torch.zeros(3)
    pixel_count = 0

    for path in image_paths:
        # Load image
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img)  # Convert to tensor (C, H, W)

        # Update sums
        pixel_sum += img_tensor.sum(dim=(1, 2))
        pixel_squared_sum += (img_tensor ** 2).sum(dim=(1, 2))
        pixel_count += img_tensor.shape[1] * img_tensor.shape[2]  # Total number of pixels per channel

    # Calculate mean and std
    mean = pixel_sum / pixel_count
    std = torch.sqrt(pixel_squared_sum / pixel_count - mean ** 2)

    return mean.tolist(), std.tolist()

# Example usage
image_dir = "seq_01/image_02/data"  # Path to your image directory
mean,std=calculate_mean_std(image_dir)
labels_dir = "seq_01/labels.txt"
digits=6
mean, std = calculate_mean_std(image_dir)

def visualize_crops_with_labels(dataset, num_samples=10):
    """
    Visualize cropped images with their corresponding labels.
    Args:
        dataset (CustomValidationDataset): Dataset containing crops and labels.
        num_samples (int): Number of samples to display.
    """
    for i in range(min(num_samples, len(dataset))):
        crops, labels = dataset[i]  # Get crops and labels for the ith frame

        # Loop through each crop and its label
        for j, (crop, label) in enumerate(zip(crops, labels)):
            plt.figure(figsize=(3, 3))
            crop_image = transforms.ToPILImage()(crop)  # Convert tensor to PIL image
            plt.imshow(crop_image)
            plt.title(f"Label: {label}")
            plt.axis("off")
            plt.show()

# Define class map for predictions
CLASS_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

# Regularization: Dropout added to ResNet
class ResNetWithDropout(nn.Module):
    def __init__(self, base_model, num_classes, dropout_rate=0.5):
        super(ResNetWithDropout, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC
        self.dropout = nn.Dropout(p=dropout_rate)  # Add dropout
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)  # New FC

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)  # Fully connected
        return x
    

class CustomValidationDataset(torch.utils.data.Dataset):
    def __init__(self, label_data, image_dir, transform=None):
        """
        Args:
            label_data (dict): Parsed label data.
            image_dir (str): Directory containing images.
            transform: Torchvision transforms for image preprocessing.
        """
        self.label_data = label_data
        self.image_dir = image_dir
        self.transform = transform
        self.frames = sorted(label_data.keys())

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image = load_frame_image(frame, self.image_dir)  # Load image
        objects = self.label_data[frame]  # Get objects for the frame

        crops = []
        labels = []
        temp = image.copy()
        for obj in objects:
            xmin, ymin, xmax, ymax = obj["bbox"]  # Get bounding box
            crop = temp.crop((xmin, ymin, xmax, ymax))  # Crop the image for each object
            if self.transform:
                crop = self.transform(crop)  # Apply transformations (resize, normalize, etc.)
            crops.append(crop)
            labels.append(obj["class"])  # Add class label
        return crops, labels  # Return cropped object images and corresponding labels



def load_frame_image(frame, image_dir):
    """
    Loads the image corresponding to a specific frame.
    Args:
        frame (int): Frame ID.
        image_dir (str): Path to the image_02/data/ directory.
    Returns:
        PIL.Image: Loaded image.
    """
    image_path = os.path.join(image_dir, f"{frame:0{digits}d}.png")  # Zero-padded frame number
    return Image.open(image_path).convert("RGB")


def parse_labels(label_file):
    """
    Parses the label file to extract frame, bounding boxes, and object classes.
    Args:
        label_file (str): Path to the labels.txt file.
    Returns:
        data (dict): Dictionary mapping frame IDs to bounding box info.
    """
    data = {}
    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        tokens = line.split()
        frame = int(tokens[0])  # Frame ID
        obj_class = tokens[2]  # Object class (Car, Pedestrian, Cyclist)
        if obj_class not in ["Car", "Pedestrian", "Cyclist"]:
            continue

        # Bounding box coordinates
        xmin, ymin, xmax, ymax = map(float, tokens[6:10])
        bbox = (int(xmin), int(ymin), int(xmax), int(ymax))

        # Add to frame's data
        if frame not in data:
            data[frame] = []
        data[frame].append({"bbox": bbox, "class": obj_class})

    return data


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)  # Normalization with ImageNet values
])

from torchvision.models import resnet50,ResNet50_Weights
import torch.nn as nn
import torch


# Load trained classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model = ResNetWithDropout(base_model=model, num_classes=len(CLASS_MAP), dropout_rate=0.5)
model.load_state_dict(torch.load("kitti_object_classifier_new_best.pth", weights_only=True))
model = model.to(device)
model.eval()

def validate_model(model, dataloader, class_map):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for crops, labels in dataloader:
            # Flatten crops and labels for multi-object handling
            crops = [item for sublist in crops for item in sublist]  # Flatten list of crops
            labels = [item for sublist in labels for item in sublist]  # Flatten list of labels

            # Batchify crops
            crops = torch.stack(crops).to(device)  # Convert list to tensor
            outputs = model(crops)  # Forward pass through the model
            _, preds = torch.max(outputs, 1)  # Get predictions

            # Convert labels to numerical values using the class_map
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend([class_map[label] for label in labels])  # Convert string labels to integers

    return all_preds, all_labels

REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

# Load data
label_data = parse_labels(labels_dir)  # Parse the label data (bounding boxes and classes)
dataset = CustomValidationDataset(label_data, image_dir, transform=transform)  # Create dataset
# Visualize crops and labels
# visualize_crops_with_labels(dataset, num_samples=1)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)  # Create dataloader

# Validate the model
predictions, ground_truths = validate_model(model, dataloader, CLASS_MAP)

# Print classification metrics
from sklearn.metrics import classification_report
print(classification_report(ground_truths, predictions, target_names=CLASS_MAP.keys(), zero_division=0))
