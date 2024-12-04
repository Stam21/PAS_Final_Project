import os
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the class map for your classes
CLASS_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

# Helper function to load frame images
def load_frame_image(frame, image_dir, digits=10):
    """
    Loads the image corresponding to a specific frame.
    """
    image_path = os.path.join(image_dir, f"{frame:0{digits}d}.png")  # Zero-padded frame number
    return Image.open(image_path).convert("RGB")

# Custom ResNet model with Dropout
class ResNetWithDropout(nn.Module):
    def __init__(self, base_model, num_classes, dropout_rate=0.5):
        super(ResNetWithDropout, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC layer
        self.dropout = nn.Dropout(p=dropout_rate)  # Add dropout layer
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)  # New FC layer

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)  # Fully connected layer
        return x

def read_bounding_boxes(file_path, image_width, image_height):
    bounding_boxes = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            lin = line.split(", ")
            parts = [
                lin[0],
                lin[1],
                " ".join(lin[next(i for i, item in enumerate(lin) if "BBox" in item):])
            ]

            if len(parts) != 3:
                print(f"Skipping malformed line: {parts}")
                continue  # Skip malformed lines
            
            try:
                frame_filename = parts[0].split(": ")[1].strip()
                track_id = int(parts[1].split(": ")[1].strip())
                bbox_str = parts[2].split(": ")[1].strip()
                bbox_values = bbox_str.strip("()")  
                bbox = tuple(map(int, bbox_values.split()))  
                xmin, ymin, xmax, ymax = bbox
                if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height:
                    print(f"Skipping invalid bounding box {bbox} for frame {frame_filename}")
                    continue  
                frame_id = int(frame_filename.split('.')[0])  
                if frame_id not in bounding_boxes:
                    bounding_boxes[frame_id] = []
                bounding_boxes[frame_id].append((track_id, "Unknown", bbox, 0.0))  
            except Exception as e:
                print(f"Error processing line: {line}. Error: {e}")
                continue  
    return bounding_boxes

def predict_bounding_box_labels(model, image_dir, bounding_boxes, class_map, frame, transform=None):
    image = load_frame_image(frame, image_dir)  
    predicted_labels = []  
    
    for track_id, class_name, bbox, depth in bounding_boxes:
        xmin, ymin, xmax, ymax = bbox  
        crop = image.crop((xmin, ymin, xmax, ymax))  
        if transform:
            crop = transform(crop)  
        crop = crop.unsqueeze(0).to(device)  
        outputs = model(crop)  
        _, pred = torch.max(outputs, 1)  
        predicted_class = list(class_map.keys())[list(class_map.values()).index(pred.item())]
        predicted_labels.append((track_id, predicted_class, bbox))  
    return predicted_labels

def calculate_mean_std(image_dir):
    transform = transforms.ToTensor()  
    pixel_sum = torch.zeros(3)  
    pixel_squared_sum = torch.zeros(3)
    pixel_count = 0
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img)  
        pixel_sum += img_tensor.sum(dim=(1, 2))
        pixel_squared_sum += (img_tensor ** 2).sum(dim=(1, 2))
        pixel_count += img_tensor.shape[1] * img_tensor.shape[2]  
    mean = pixel_sum / pixel_count
    std = torch.sqrt(pixel_squared_sum / pixel_count - mean ** 2)
    return mean.tolist(), std.tolist()

# Define image transformations
mean, std = calculate_mean_std("seq_03/image_02/data")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.DEFAULT)  
model = ResNetWithDropout(base_model=model, num_classes=len(CLASS_MAP), dropout_rate=0.5)
model.load_state_dict(torch.load("kitti_object_classifier_new_best.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# Load the first image to get dimensions
sample_image = load_frame_image(1, "seq_03/image_02/data")
image_width, image_height = sample_image.size  

# Read bounding boxes
bounding_boxes_file = "bounding_boxes.txt"
bounding_boxes_data = read_bounding_boxes(bounding_boxes_file, image_width, image_height)

# Animation setup
fig, ax = plt.subplots(figsize=(12, 8))  # Create the figure
ax.axis("off")
im_display = ax.imshow(sample_image)

# Make the window fullscreen
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

def update(frame):
    image = load_frame_image(frame, "seq_03/image_02/data")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=24)
    bounding_boxes = bounding_boxes_data.get(frame, [])
    predicted_labels = predict_bounding_box_labels(model, "seq_03/image_02/data", bounding_boxes, CLASS_MAP, frame, transform)
    for track_id, label, bbox in predicted_labels:
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=3)
        draw.text((xmin, ymin), f"{label}", fill="yellow", font=font)
    im_display.set_data(image)
    return [im_display]

ani = FuncAnimation(fig, update, frames=sorted(bounding_boxes_data.keys()), interval=100, blit=True)
plt.show()
