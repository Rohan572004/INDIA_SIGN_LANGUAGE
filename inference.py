import torch
import mediapipe as mp
import cv2
import numpy as np
from torchvision import transforms
from models.mobilenet_model import get_model
from utils.data_loader import get_data_loaders

# Configuration
MODEL_PATH = 'D:/sign_language_mobilenet.pth'
IMG_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)
    
def preprocess_hand_crop(hand_crop):
    """Preprocess cropped hand area for model input"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(hand_crop).unsqueeze(0)

def main():
    # Load model
    model = get_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Get class names
    _, _, classes = get_data_loaders('DataISL')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box coordinates
                h, w, _ = frame.shape
                x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
                x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
                y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
                y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Crop the frame to the bounding box
                hand_crop = frame[y_min:y_max, x_min:x_max]
                if hand_crop.size != 0:
                    # Resize the cropped hand area to the input size for the model
                    hand_crop_resized = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
                    input_tensor = preprocess_frame(hand_crop_resized)
                    input_tensor = input_tensor.to(DEVICE)
                    
                    # Make prediction
                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        prediction = classes[predicted.item()]
                
                # Display prediction
                cv2.putText(frame, f'Prediction: {prediction}', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Sign Language Recognition', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
