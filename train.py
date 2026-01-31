import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from dataset import get_dataloaders
from models.resnet_model import DogEmotionResNet
from utils import evaluate, train_accuracy
from config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH

def train():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    train_loader, val_loader = get_dataloaders()
    
    
    print(f"STAGE 1: Training the Head (Backbone Frozen) for 10 Epochs...")
    model = DogEmotionResNet(freeze_backbone=True).to(DEVICE)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_acc = 0.0

    
    for epoch in range(10):
        run_epoch(model, train_loader, val_loader, optimizer, criterion, epoch, 10, 
                  train_losses, train_accuracies, val_losses, val_accuracies, DEVICE)
        
        
        current_val_acc = val_accuracies[-1]
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    
    print(f"\n STAGE 2: Unfreezing all layers and lowering Learning Rate...")
    
    
    for param in model.parameters():
        param.requires_grad = True
        
    
    fine_tune_lr = 1e-5 
    optimizer = Adam(model.parameters(), lr=fine_tune_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    
    for epoch in range(10, EPOCHS):
        run_epoch(model, train_loader, val_loader, optimizer, criterion, epoch, EPOCHS, 
                  train_losses, train_accuracies, val_losses, val_accuracies, DEVICE)
        
        
        scheduler.step(val_losses[-1])
        current_val_acc = val_accuracies[-1]
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Stage 2 Improvement: {best_val_acc*100:.2f}%")

    print("\n All Training Complete!")
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

def run_epoch(model, train_loader, val_loader, optimizer, criterion, epoch, total_epochs, t_loss, t_acc, v_loss, v_acc, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{total_epochs}]")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images); loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        running_loss += loss.item()
    
    
    avg_t_loss = running_loss / len(train_loader)
    train_acc = train_accuracy(model, train_loader, device)
    val_acc, val_loss = evaluate(model, val_loader, device)
    
    t_loss.append(avg_t_loss); t_acc.append(train_acc)
    v_loss.append(val_loss); v_acc.append(val_acc)
    print(f"  > Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(train_losses); plt.plot(val_losses); plt.title('Loss'); plt.legend(['Train', 'Val'])
    plt.subplot(1, 2, 2); plt.plot(train_accuracies); plt.plot(val_accuracies); plt.title('Accuracy'); plt.legend(['Train', 'Val'])
    plt.show()

if __name__ == "__main__":
    train()