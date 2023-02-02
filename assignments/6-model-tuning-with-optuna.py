import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Resize
import matplotlib.pyplot as plt

import optuna
from optuna.trial import TrialState

DEVICE = torch.device("cpu")

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

class NeuralNetwork(nn.Module):
    def __init__(self, trial):
        super().__init__()
        self.flatten = nn.Flatten()
        ### START CODE HERE ###
        n_hidden_layers = trial.suggest_int("n_layers", 1, 3)
        modules = []

        in_features = 32*32*3

        for i in range(n_hidden_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 500)
            modules.append(nn.Linear(in_features, out_features))
            modules.append(nn.ReLU())

            in_features = out_features

        modules.append(nn.Linear(in_features, 10))
        self.linear_relu_stack = nn.Sequential(*modules)


    def forward(self, inputs):
        x = self.flatten(inputs)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        """
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        """

# Define a test function to evaluate model performance
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, accuracy = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            #test_loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy /= size
    #print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return accuracy



def objective(trial):

    # Generate the model.
    model = NeuralNetwork(trial).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset.
    train_batch_size = trial.suggest_int("train_bs", 1, 10000)
    test_batch_size = trial.suggest_int("test_bs", 1, 5000)
    train_dataloader = DataLoader(training_data, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size)

    EPOCHS = 10

    # Training of the model
    for epoch in range(EPOCHS):
        #print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        accuracy = test(test_dataloader, model, loss_fn)
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy



if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10000, timeout=72000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))