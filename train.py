import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.data_loader import get_data_loaders
from models.mobilenet_model import get_model
import os


BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005
IMG_SIZE = 224
DATA_DIR = 'DataISL'
MODEL_SAVE_PATH = 'D:/sign_language_mobilenet.pth'
PATIENCE = 5

import matplotlib.pyplot as plt

def train():
    # Initialize metric trackers
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
  
    train_loader, test_loader, classes = get_data_loaders(DATA_DIR, BATCH_SIZE, IMG_SIZE, augment=True, 
        augment_params={
            'rotation_range': 30,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        })
    print(f"Number of classes: {len(classes)}")
   
    model = get_model(device)
    # Calculate class weights
    class_counts = [len([x for x in train_loader.dataset.samples if x[1] == i]) for i in range(len(classes))]
    class_weights = torch.tensor([1.0 - (count / sum(class_counts)) for count in class_counts], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
   
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
           
            optimizer.zero_grad()
            
           
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
           
            loss.backward()
            optimizer.step()
            
         
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:  
                epoch_acc = 100. * correct / total
                train_accuracies.append(epoch_acc)
                train_losses.append(running_loss/100)
                
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}, Acc: {epoch_acc:.2f}%')
                running_loss = 0.0
        
       
        scheduler.step()
        
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(test_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
        
        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'Model improved and saved to {MODEL_SAVE_PATH}')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    print('Training complete')
    
    # Save metrics to file
    import json
    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    train()
