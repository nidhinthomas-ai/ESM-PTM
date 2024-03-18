import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNNModel(nn.Module):
    """
    A convolutional neural network model for binary classification tasks.
    """

    def __init__(self, input_size):
        """
        Initializes the CNN model with specified input size and architecture parameters.

        Args:
            input_size (int): The number of input features.
        """
        super(CNNModel, self).__init__()
        # Architecture parameters
        filters = 128
        kernels = 5
        dense_layers1 = 64
        dense_layers2 = 32
        dropout_rate = 0.2

        # Model layers
        self.conv1 = nn.Conv1d(input_size, filters, kernel_size=kernels, padding='same')
        self.conv1_bn = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filters, dense_layers1)
        self.fc2 = nn.Linear(dense_layers1, dense_layers2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(dense_layers2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the CNN model.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the network.
        """
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def cnn_train(X_train, Y_train, X_valid, Y_valid):
    """
    Trains the CNN model on the training dataset and evaluates on the validation dataset.

    Args:
        X_train (Tensor): Training features.
        Y_train (Tensor): Training labels.
        X_valid (Tensor): Validation features.
        Y_valid (Tensor): Validation labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(X_train.shape[2]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert datasets to TensorDatasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(Y_valid, dtype=torch.float32))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    # Training loop
    for epoch in range(100):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}: Training Loss: {loss.item()}')

if __name__ == "__main__":

    X_train, Y_train, X_valid, Y_valid = None, None, None, None
    cnn_train(X_train, Y_train, X_valid, Y_valid)
