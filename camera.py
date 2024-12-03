import torch
import cv2
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class CNN6Conv6FC(nn.Module):
    def __init__(self, num_classes, num_blocks):
        super(CNN6Conv6FC, self).__init__()
        self.num_classes = num_classes
        self.conv0 = nn.Conv2d(in_channels=1,  out_channels=64, kernel_size=3, stride=1, padding=1)

        preserve = []
        collapse = []
        for i in range(num_blocks):
            sq = nn.Sequential(
                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
                     nn.BatchNorm2d(64),
                     nn.LeakyReLU(negative_slope=0.01, inplace=True))
            preserve.append(sq)
            collapse.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=1),
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=0))
            )

            
        self.preserve = nn.ModuleList(preserve)
        self.collapse = nn.ModuleList(collapse)

        self.classifier = nn.Sequential(
            nn.LazyLinear(1024),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(64, num_classes) 
        )
        
    def forward(self, x):
        x = self.conv0(x)

        for pr, cl in zip(self.preserve, self.collapse):
            x = pr(x) + x
            x = cl(x)

        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        
        return x


model = CNN6Conv6FC(num_classes=5, num_blocks=4)
state_dict = torch.load(r'./model_2.pt')
model.load_state_dict(state_dict)
model.eval() 

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Grayscale(), 
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

cap = cv2.VideoCapture(r'./videoTest.mp4')

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