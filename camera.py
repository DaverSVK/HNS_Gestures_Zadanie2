import torch
import cv2
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class CNN6Conv6FC(nn.Module):
    def __init__(self, num_classes):
        super(CNN6Conv6FC, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)  
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, num_classes)  
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        x = self.fc6(x)
        
        return x

model = CNN6Conv6FC(num_classes=5)
state_dict = torch.load(r'C:\Users\david\Desktop\Code\DNN\Projekt_2\BestModel2.pt')
model.load_state_dict(state_dict)
model.eval() 

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Grayscale(), 
    transforms.Normalize(mean=[0.485], std=[0.229]),
])


video_path = 'path/to/your/video.mp4' 
cap = cv2.VideoCapture(r'C:\Users\david\Desktop\Code\DNN\Projekt_2\videoTest.mp4')

if not cap.isOpened():
    print("Error: Unable to open the video file")
    exit()

print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to capture frame")
        break

    # Transform frame
    transformed_frame = preprocess(frame).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(transformed_frame)

    predicted_class = torch.argmax(outputs, dim=1).item()

    # Map prediction to label
    class_labels = {0: 'Neutral', 1: 'OcclusionEyes', 2: 'OcclusionMouth', 3: 'OpenMouth', 4: 'Smile'} 
    label = class_labels.get(predicted_class, 'Unknown')

    # Add label to the frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert transformed tensor back to NumPy array for display
    transformed_image = transformed_frame.squeeze(0).numpy()
    transformed_image = transformed_image[0]  # Grayscale single channel
    transformed_image = (transformed_image * 0.229 + 0.485) * 255  # De-normalize
    transformed_image = transformed_image.astype(np.uint8)

    # Show transformed image
    cv2.imshow('Transformed Image', transformed_image)

    # Show original frame with prediction
    cv2.imshow('Object Detection', frame)

    # Limit to 10 fps

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()