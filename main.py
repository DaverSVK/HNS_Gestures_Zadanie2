import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import models, datasets
import torchvision.transforms as transforms
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST 
from torch.utils.tensorboard import SummaryWriter
import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
import torch.optim as optim
import cnnAndMlp

from torch.utils.tensorboard import SummaryWriter


print(f"Is GPU available? {torch.cuda.is_available()}")
print(f"Number of available devices: {torch.cuda.device_count()}")
print(f"Index of current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Oznacenie datasetu
data_flag = 'pneumoniamnist'
download = True

# Nastavenie siete
NUM_EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 0.001
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])
# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)
pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

writer = SummaryWriter('runs/experiment_55')

model = cnnAndMlp.CustomCNN2(in_channels=n_channels, num_classes=n_classes)
    
# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
    
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=lr)


# Function to plot the confusion matrix
import seaborn as sns
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Class names for binary classification
class_names = ['Normal', 'Pneumonia']

acc_history = []
loss_history = []
final_labels = []
final_predicted = []    
# train
def trainingFnc():
    for epoch in range(NUM_EPOCHS):
        running_corrects = 0 
        total = 0
        epoch_loss = 0
        n_batches = len(train_dataset) // BATCH_SIZE
        model.train()
        
        for inputs, targets in tqdm(train_loader):
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            epoch_loss += loss.item() 
            
            _, preds = torch.max(outputs, 1)  # Get the predicted class (argmax)
            running_corrects += torch.sum(preds == targets).item()  # Compare with actual labels
            total += targets.size(0)  # Update total number of samples
            
            loss.backward()
            optimizer.step()
            writer.add_scalar('training loss (batch)', loss.item(), epoch * len(train_loader) + i)

            avg_epoch_loss = epoch_loss / len(train_loader)
            accuracy_train = running_corrects / total
            writer.add_scalar('training loss (epoch)', avg_epoch_loss, epoch)
            writer.add_scalar('training accuracy (epoch)', accuracy_train, epoch)

    writer.add_hparams(
    {
    'optimizer': optimizer.__class__.__name__,
    'lr': LEARNING_RATE, 
    'batch_size': BATCH_SIZE
    },
    {
    'hparam/train/accuracy': accuracy_train,
    }
    )
    writer.close()

trainingFnc()


def test(split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        # Convert y_score to predicted labels if classification task
        if task == 'multi-label, binary-class':
            y_pred = (y_score > 0.5).astype(int)  # Threshold for binary classification
        else:
            y_pred = np.argmax(y_score, axis=-1)
        
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)
    
        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, class_names)
        
print('==> Evaluating CNN0...')
test('train')
test('test')

PATH = "./model.pt"
torch.save(model.state_dict(), PATH)

# model = cnnAndMlp.CustomCNN1(in_channels=n_channels, num_classes=n_classes)
# trainingFnc()
# print('==> Evaluating CNN1...')
# test('train')
# test('test')
# model = cnnAndMlp.CustomCNN2(in_channels=n_channels, num_classes=n_classes)
# trainingFnc()
# print('==> Evaluating CNN2...')
# test('train')
# test('test')
# model = cnnAndMlp.CustomMLP(in_channels=n_channels, num_classes=n_classes)
# trainingFnc()
# print('==> Evaluating MLP...')
# test('train')
# test('test')



def visualize_filters(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            filters = module.weight.data.cpu()

            # Normalize the filter values to 0-1 for better visualization
            filters = (filters - filters.min()) / (filters.max() - filters.min())

            # Plot each filter as an image
            num_filters = filters.shape[0]  # Number of filters
            fig, axs = plt.subplots(1, num_filters, figsize=(15, 15))

            for i in range(num_filters):
                axs[i].imshow(filters[i, 0, :, :], cmap='gray')
                axs[i].axis('off')
            
            plt.show()
            break  # Only visualize the first convolutional layer

visualize_filters(model)