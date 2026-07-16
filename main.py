#copied from mnist (https://github.com/manaspaldhe12/mnist) but switching dataset to imagenet

import csv
import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from datetime import datetime
import torch.optim as optim

import argparse

import logging

# Configure the logger to save to a local file
logging.basicConfig(
    filename='app.log',          # The name of your local log file
    filemode='a',                # 'a' appends data; 'w' overwrites the file each run
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO           # Minimum severity level to capture
)



# =====================================================================
# COMMAND-LINE INTERFACE CONFIGURATION
# =====================================================================
parser = argparse.ArgumentParser(description="Train and evaluate a simple MNIST NN.")

# Flag that defaults to True. Passing '--no_latest' flips it to False.
parser.add_argument(
    '--no_latest', 
    dest='use_latest_model', 
    action='store_false', 
    help="Do not load the latest trained model; start fresh."
)
parser.set_defaults(use_latest_model=True)

# Flag that defaults to False. Passing '--train' flips it to True.
parser.add_argument(
    '--train', 
    dest='train_model', 
    action='store_true', 
    help="Run the training loop phase."
)
parser.set_defaults(train_model=False)

# Parse the arguments from the terminal execution command
args = parser.parse_args()

# Access values via args.use_latest_model and args.train_model
USE_LATEST_MODEL = args.use_latest_model
TRAIN_MODEL = args.train_model

# resize because the image sizes are different...
imagenet_transforms = transforms.Compose([
    transforms.Resize((256,256))
])

# Load the Training Data
# Note: You must have manually downloaded and extracted ImageNet to './data/imagenet'
train_dataset = torchvision.datasets.Imagenette( # switched to this for autodownload feature.. (gemini recommended)
    root='./data', 
    split='train', 
    download=True,       # This works automatically!
    transform=imagenet_transforms
)

# Download and load the Validation Data
test_dataset = torchvision.datasets.Imagenette(
    root='./data', 
    split='val', 
    download=True,
    transform=imagenet_transforms
)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

# Define the network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function
        self.relu = nn.ReLU()
        # Second fully connected layer (output layer)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_size, output_size)
        # then a sigmoid layer
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x):
        # Pass input through the first layer
        x = self.fc1(x)
        # Apply the ReLU activation
        x = self.relu(x)
        # Pass through the fc2 layer
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        # sigmoid
        x = self.sigmoid(x)	
        return x


model = SimpleNN(input_size=256*256, hidden_size=18, output_size=1000)

model_loaded = False
if USE_LATEST_MODEL:
    # Look for any files matching the pattern
    saved_models = glob.glob("model_*.pth")
    if saved_models:
        # Sort files by modification time to grab the absolute latest one
        latest_model_file = max(saved_models, key=os.path.getmtime)
        logging.info(f"--> Found existing models. Loading weights from: {latest_model_file}")
        model.load_state_dict(torch.load(latest_model_file))
        model_loaded = True
    else:
        logging.info("--> USE_LATEST_MODEL is True, but no 'model_*.pth' files were found. Starting from scratch.")

logging.info(f"Network Architecture:\n {model}")

criterion = nn.MSELoss(reduction='sum')

if TRAIN_MODEL or not model_loaded:
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(1000):
        running_loss = 0.0
        for images, labels in train_loader:
            # Flatten images from (batch_size, 1, 28, 28) to (batch_size, 784)
            # default is not a vector, so I was having issues.
            images = images.reshape(-1, 28*28)
            images = images.float() 
            optimizer.zero_grad()
            outputs = model(images)
            # default is not one hot in mnist, so I got a warning.
            one_hot_labels = nn.functional.one_hot(labels, num_classes=1000).float()
            loss = criterion(outputs, one_hot_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #if epoch%100 == 0:
        #   kear
        # I was about to do rate change.. but thanks to AI realized that adam is already adaptive...
        # what should I experiment with then... what will be different from mnist if I just copy paste....
        # things like image transformations etc I could try with mnist too (not rotation lol else 6 becomes 9)
        # but at least resnet or conv layers...
        # let me think what I can do with Imageneet that I cannot with mnist in terms of learning...
            
        logging.info(f"Epoch {epoch+1} finished. Avg Loss: {running_loss/len(train_loader):.4f}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_{timestamp}.pth"
    # Save the state dict
    torch.save(model.state_dict(), filename)


model.eval()  # Disables dropout and batch normalization updates
total_loss = 0.0
correct_predictions = 0
total_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        images = images.float() 
        outputs = model(images)
        one_hot_labels = nn.functional.one_hot(labels, num_classes=1000).float()
        loss = criterion(outputs, one_hot_labels)
        total_loss += loss.item()
        # Calculate accuracy (assuming a classification task)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)


average_test_loss = total_loss / total_samples
test_accuracy = correct_predictions / total_samples

logging.info(f"Test Loss: {average_test_loss:.4f}")
logging.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Prepare the data row
timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
csv_file = 'results.csv'
file_exists = os.path.isfile(csv_file)

# Append the metrics to the CSV file
with open(csv_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    
    # Write the header line only if the file is being newly created
    if not file_exists:
        writer.writerow(['Timestamp', 'Model File', 'Avg Train Loss', 'Test Loss', 'Test Accuracy (%)'])
        
    # Append the results row
    writer.writerow([
        timestamp_str,
        filename if (TRAIN_MODEL or not model_loaded) else latest_model_file,
        f"{running_loss/len(train_loader):.4f}" if (TRAIN_MODEL or not model_loaded) else "N/A",
        f"{average_test_loss:.4f}",
        f"{test_accuracy * 100:.2f}%"
    ])

logging.info(f"Successfully appended execution results to {csv_file}")
