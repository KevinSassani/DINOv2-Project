
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoImageProcessor, Dinov2ForImageClassification
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from glob import glob
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
 
def save_checkpoint(model,history,epoch):
    path = 'checkpoints/dinov2_vits14_lc_checkpoint_'+str(epoch)+'.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,  # inludes loss, acc, val_loss, val_acc
        'epoch': epoch
    }, path)

def load_checkpoint(path):
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    # freeze all weights in the dinov2_vits14_lc_copy except the new head
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the new head
    for param in model.linear_head.parameters():
        param.requires_grad = True
    model.eval()
    history = checkpoint['history']
    return model, epoch, history

def validation_fn(model,loss_function,val_loader):
    running_loss = 0 
    total = 0
    correct = 0
    batches = len(val_loader)
    with torch.no_grad():
        batch = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
	  
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = loss_function(outputs,labels)
            running_loss += loss.item()
         
    accuracy = correct / total   
  
    return accuracy, running_loss/batches 

batch_size = 128
train_val_split = 0.05

# Set seed for reproducibility
torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize images to 224x224
    transforms.ToTensor(),         # Convert images to PyTorch tensors and normalize to [0, 1]
])
# Load Data
train_data = torchvision.datasets.Food101(root='food101', split='train', download=True, transform=transform)
test_data = torchvision.		datasets.Food101(root='food101', split='test', download=True, transform=transform)

# Split train_data to validation set and training set
val_size = int(len(train_data)*train_val_split)
train_size = len(train_data)-val_size

train_data, val_data = random_split(train_data, [train_size, val_size])
#print(val_data)
print('Size of Training Data: ', len(train_data))
print('Size of Validation Data: ', len(val_data))
print('Size of Test Data: ', len(test_data))

# Prepare loaders
#train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

create_model = False

if create_model:
    dinov2_vits14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')

    # Inspect the current classification head
    existing_head = dinov2_vits14_lc.linear_head

    # Get the shape of the existing head
    input_features = existing_head.in_features
    output_features = existing_head.out_features

    # Initialize a new classification head with random weights
    new_head = nn.Linear(input_features, output_features)

    print(f"New head initialized with shape: {input_features} -> {output_features}")
    import copy

    # Deep copy the model
    dinov2_vits14_lc_original = copy.deepcopy(dinov2_vits14_lc)
    dinov2_vits14_lc.linear_head = new_head


    # freeze all weights in the dinov2_vits14_lc_copy except the new head
    for param in dinov2_vits14_lc.parameters():
        param.requires_grad = False

    # unfreeze the new head
    for param in dinov2_vits14_lc.linear_head.parameters():
        param.requires_grad = True
    history = {
        'loss': np.array(()),
        'acc': np.array(()),
        'val_loss': np.array(()),
        'val_acc': np.array(())
    }
    epoch = 0
    model = dinov2_vits14_lc
    save_checkpoint(model,history,epoch)
else:
    # Choose the latest checkpoint here
    path = 'checkpoints/dinov2_vits14_lc_checkpoint_15.pth'
    print('Initializing from: ' + path)
    model, epoch, history = load_checkpoint(path)

# Train the model


model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.linear_head.parameters(), lr=5e-4, weight_decay=1e-4)
end_epoch = 100
start_epoch = 15
validation = True

for epoch in range(start_epoch+1, end_epoch+1):
    running_loss = 0.0
    correct = 0
    total = 0
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # Shuffle samples each epoch
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{end_epoch}")
    for batch_idx, (images, labels) in progress_bar:
        optimizer.zero_grad()
        # input = processor(images=batch[0], return_tensors="pt")
        input = processor(images=images, return_tensors="pt", padding=True).to(device)
        # input = processor(images=batch[0], return_tensors="pt").to(device)
        labels = labels.to(device)
        output = model(input["pixel_values"])
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        # Update running loss and accuracy
        running_loss += (loss.item()-running_loss)/(batch_idx+1)
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        acc = correct/total

        # Update progress bar with loss
        progress_bar.set_postfix(loss=running_loss,accuracy=f"{100*acc:.2f}%")
             
    
    if validation:
        val_acc, val_loss = validation_fn(model,loss_function,val_loader)
        print('val_acc: ', val_acc,' val_loss: ', val_loss)
    else:
        val_loss = -1
        val_acc = -1
    # Save history
    history = {
        'loss': np.append(history['loss'], running_loss),
        'acc': np.append(history['acc'], acc),
        'val_loss': np.append(history['val_loss'], val_loss),
        'val_acc': np.append(history['val_acc'], val_acc)
    }
    if epoch % 5  == 0:
        save_checkpoint(model, history, epoch)
