import matplotlib.pyplot as plt
import json
import os

def plot_metrics():
    # Check if metrics file exists
    if not os.path.exists('training_metrics.json'):
        print("No training metrics found. Please run train.py first to generate metrics.")
        return
    
    try:
        # Load metrics from file
        with open('training_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        # Plot training and validation metrics
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(metrics['train_losses'], label='Train Loss')
        plt.plot(metrics['val_losses'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(metrics['train_accuracies'], label='Train Accuracy')
        plt.plot(metrics['val_accuracies'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()
        
    except Exception as e:
        print(f"Error loading or plotting metrics: {str(e)}")

if __name__ == '__main__':
    plot_metrics()
