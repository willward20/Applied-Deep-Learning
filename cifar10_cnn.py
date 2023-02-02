################################################################
# This program builds and trains a convolutional neural network
# using Pytorch and the CIFAR10 dataset.
################################################################


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Resize
import matplotlib.pyplot as plt


# Designate processing unit for CNN training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Create class for CNN architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # these lines just define the parameters and output of each nn layer
        self.conv1 = nn.Conv2d(3, 48, 3) # (color channel size, output channel size, kernel size (5x5))
        self.pool = nn.MaxPool2d(2, 2) # (output size of 2, stride size of 2)
        self.conv2 = nn.Conv2d(48, 192, 2) # (input size same as conv1 output, output size conv2, kernel (5x5))
        self.conv3 = nn.Conv2d(192, 128, 2)
        self.fc1 = nn.Linear(128*3*3, 120) # 16*5*5 is the flattened output size of the output image from last pooling layer (after conv2)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 is for 10 classes

    def forward(self, x): # this defines the order that layers are executed
        x = self.pool(F.relu(self.conv1(x))) # conv1 --> relu activation --> pool
        x = self.pool(F.relu(self.conv2(x))) # conv2 --> relu activation --> pool
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flattens tensor, except the batch dimension
        x = F.relu(self.fc1(x)) # fc1 --> relu activation
        x = F.relu(self.fc2(x)) # fc2 --> relu activation
        x = self.fc3(x) # no activation function here
        return x


# Define a CNN training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_losses = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            train_losses.append(loss)
    return train_losses


# Define a CNN testing function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, accuracy = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy /= size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")




#############################################
# 1. Download CIFAR10 Datasets
#############################################

# Download training data from open datasets.
training_data = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.CIFAR10(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)

# Investigate datasets
num_train = len(training_data)
res_train = training_data[0][0].shape
num_test = len(test_data)
res_test = test_data[0][0].shape
print(f"number of training examples: {num_train}")
print(f"training image resolution (CHANNEL, HEIGHT, WIDTH): {res_train}")
print(f"number of test examples: {num_test}")
print(f"test image resolution (CHANNEL, HEIGHT, WIDTH): {res_test}")

# Create a dictionary for image label names
labels_map = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}


################################################
# 2. Test to make sure the network is functional
################################################

model = ConvNet().to(device)    # instantiate the model
X = torch.rand(1, 3, 32, 32, device=device) # create random 3x32x32 "image"
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class probability: {pred_probab}")
print(f"Predicted class: {y_pred}: {labels_map[int(y_pred)]}")


################################################
# 3. Prepare to train the NN
################################################

cifar_model = ConvNet().to(device)  # instantiate the model
train_dataloader = DataLoader(training_data, batch_size=100)    # load the train data in batches of 100 images
test_dataloader = DataLoader(test_data, batch_size=1000)    # load the test data in batches of 1000 images
loss_fn = nn.CrossEntropyLoss() # use the Cross Entropy loss function
optimizer = torch.optim.Adam(cifar_model.parameters(), lr= 0.001) # use the Adam optimizer with a learning rate of 0.001
epochs = 10 # repeat the train and test loop 10 times


################################################
# 4. Train the CNN image classifier 
################################################

sum_train_losses = []   # array to track train loss after each epoch
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_losses = train(train_dataloader, cifar_model, loss_fn, optimizer)
    sum_train_losses += train_losses
    test(test_dataloader, cifar_model, loss_fn)
print("Done!")
plt.plot(sum_train_losses) # Plot train losses
plt.show()


################################################
# 5. Test the model on an image from the datset
################################################

cifar_model.eval()
x, y = test_data[20][0], test_data[20][1]
x = x.unsqueeze(dim=0)
with torch.no_grad():
    pred = cifar_model(x.to(device))
    predicted, actual = labels_map[int(pred[0].argmax(0))], labels_map[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
