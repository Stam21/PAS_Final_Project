import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# Map class names to numeric labels
CLASS_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


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

image_dirr = "./Kitti/data_object_image_2/training/image_2"
label_dirr = "./Kitti/data_object_label_2/training/label_2"
mean,std = calculate_mean_std(image_dirr)

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
    
class KittiMultiObjectDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, max_images=None, startingPoint=0):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        if max_images:
            self.image_files = self.image_files[startingPoint:max_images]  # Limit to first 1000 images
            self.label_files = self.label_files[startingPoint:max_images]
        self.transform = transform
        self.samples = self._create_samples()
    
    def _create_samples(self):
        samples = []
        for image_file, label_file in zip(self.image_files, self.label_files):
            image_path = os.path.join(self.image_dir, image_file)
            label_path = os.path.join(self.label_dir, label_file)
            
            # Load image and labels
            image = Image.open(image_path).convert("RGB")
            with open(label_path, "r") as f:
                lines = f.readlines()

            bboxes = []
            labels = []

            # Parse labels to get bounding boxes and class labels
            for line in lines:
                tokens = line.split()
                class_name = tokens[0]
                if class_name in CLASS_MAP:
                    xmin, ymin, xmax, ymax = map(float, tokens[4:8])
                    bboxes.append((xmin, ymin, xmax, ymax))
                    labels.append(CLASS_MAP[class_name])
            
            # Create crop samples for each bounding box
            for bbox, label in zip(bboxes, labels):
                xmin, ymin, xmax, ymax = map(int, bbox)
                crop = image.crop((xmin, ymin, xmax, ymax))
                if self.transform:
                    crop = self.transform(crop)  # Apply transformation to the crop
                samples.append((crop, label))  # Append the crop and label as a sample
        
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        crop, label = self.samples[idx]
        return crop, label

    def __del__(self):
        # Explicitly clear memory for samples and file lists
        del self.samples
        del self.image_files
        del self.label_files
        print("KittiMultiObjectDataset object deleted and memory cleared.")
    
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)  # Normalization with ImageNet values
])

def custom_collate_fn(batch):
    flat_images = []
    flat_labels = []
    
    for crops, labels in batch:
        flat_images.append(crops)  # Add each crop
        flat_labels.append(labels)  # Add corresponding label
    
    # Stack all images and labels into tensors
    images_tensor = torch.stack(flat_images)  # Stack all crops into a single tensor
    labels_tensor = torch.tensor(flat_labels)  # Convert labels to a tensor
    
    return images_tensor, labels_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model = ResNetWithDropout(base_model=model, num_classes=len(CLASS_MAP), dropout_rate=0.4)
model = model.to(device)

# Optimizer with Weight Decay (L2 Regularization)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
# Cross-Entropy Loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.GradScaler()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Early Stopping Parameters
# Early stopping set that high it won't be used!
early_stopping_patience = 100 # Stop after 100 epochs without improvement
best_loss = float('inf')
epochs_since_improvement = 0

def train_with_epochs(dataset, model, criterion, optimizer, scheduler, epochs=5):
    # Create the DataLoader for the entire dataset (no validation here)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    global best_loss, epochs_since_improvement
    # Train for the specified number of epochs
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        correct = 0
        total = 0

        # Create a tqdm progress bar to show training progress
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision
            with torch.autocast("cuda"):  
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backpropagation with scaled gradients
            scaler.scale(loss).backward()  # Scale the loss and compute gradients
            scaler.step(optimizer)         # Perform the optimizer step
            scaler.update() 

            # Accumulate metrics
            train_loss += loss.detach().item()  # Use .detach() to ensure no gradient tracking
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Update progress bar description with current training metrics
            progress_bar.set_postfix(loss=train_loss / (progress_bar.n + 1), accuracy=100.0 * correct / total)

        # Calculate average loss and accuracy for the epoch
        train_loss /= len(train_loader)
        train_accuracy = 100.0 * correct / total

        # Step the scheduler based on the training loss
        scheduler.step()  # Update the learning rate based on training loss

        # Early stopping: check if loss improved
        if train_loss < best_loss:
            best_loss = train_loss
            epochs_since_improvement = 0
            print(f"Improvement detected, saving the model...")
            torch.save(model.state_dict(), "best_model.pth")  # Save the best model
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epochs.")
        
        # If no improvement for 'patience' epochs, stop training
        if epochs_since_improvement >= early_stopping_patience:
            print("Early stopping triggered, no improvement for several epochs.")
            break
        
        # Print training progress after each epoch
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    
    print("\n--- Training Completed ---")
    print(f"Final Training Accuracy: {train_accuracy:.2f}%")


# Define the number of total images to load (7480) and the batch increment for each iteration
total_images = 7480
increment = 500  # Increment by 1000 images per iteration, adjust as needed

# Loop over and progressively load more images
start_point = 0

while start_point < total_images:
    # Ensure that we don't exceed total_images
    max_images = min(start_point + increment, total_images)

    # Create the dataset for the current iteration with the dynamic range
    dataset = KittiMultiObjectDataset(
        image_dir=image_dirr, 
        label_dir=label_dirr, 
        transform=transform,
        max_images=max_images,  # Load up to max_images in the current iteration
        startingPoint=start_point
    )

    # Train the model with the current dataset
    train_with_epochs(dataset, model, criterion, optimizer, scheduler, epochs=5)
    
    # Update the starting point for the next iteration
    start_point += increment
    del dataset  # Clear memory after each iteration

# Save the final model after all training phases
torch.save(model.state_dict(), "kitti_object_classifier.pth")

