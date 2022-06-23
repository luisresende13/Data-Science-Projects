import json
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np; np.random.seed(159)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from IPython.display import clear_output as co
from sklearn.metrics import accuracy_score as acc, classification_report as cr, precision_score as ps, r2_score as r2, recall_score as rs

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Define the class XOR_Data
class Load_Data(Dataset):
    # N_s is the size of the dataset
    def __init__(self, x, y):        
        # Create a N_s by 2 array for the X values representing the coordinates
        self.x = torch.Tensor(x)
        # Create a N_s by 1 array for the class the X value belongs to
        self.y = torch.Tensor(y)
        self.len = y.size
    # Getter
    def __getitem__(self, index):    
        return self.x[index], self.y[index]
    # Get Length
    def __len__(self):
        return self.len

class Deep_NN(nn.Module):
    # Given a list of integers, Layers, we create layers of the neural network where each integer in Layers corresponds to the layers number of neurons
    def __init__(self, Layers):        
        super(Deep_NN, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers[:-1], Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))  
    # Puts the X value through each layer of the neural network while using the RELU activation function in between. The final output is put through Sigmoid.
    def forward(self, x, activation=None):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                if activation is None: x = linear_transform(x)
                elif activation=='relu': x = F.relu(linear_transform(x))    
                elif activation=='sigmoid': x = torch.sigmoid(linear_transform(x))    
            else: x = torch.sigmoid(linear_transform(x))
        return x
    
from sklearn.metrics import accuracy_score as acc_scr
def accuracy(model, dataset):
    return acc_scr( dataset.y.view(-1).numpy(), ((model(dataset.x)>0.5)+0).numpy() )

# Function to Train the Model

def train(train_data, model, criterion, train_loader, optimizer, activation='sigmoid', epochs=5, test_data=None):
    
    if len(np.unique(train_data.y)) > 2: multi_class=True
    else: multi_class = False
    COST = []; ACC = []; test_ACC = []
    for epoch in range(epochs):
        
        if epoch in range(9, epochs, 10): 
            co(wait=True); print(f'Epoch: {epoch+1}/{epochs}')        
        
        total_loss=0
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x, activation=activation)
            if multi_class: y = torch.LongTensor(y.numpy().reshape(-1))
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        ACC.append(accuracy(model, train_data))
        COST.append(total_loss)
        if type(test_data)!=type(None): test_ACC.append(accuracy(model, test_data))    
    
    return COST, ACC, test_ACC

def Build_Layers(Dataset, hidden_layers=[5,5,5]):
    n_categories = len( np.unique(Dataset.y) )
    input_layer = Dataset.x.shape[1] # n_features
    output_layer = n_categories if n_categories!=2 else 1
    return [input_layer] + hidden_layers + [output_layer]

def execute_neural_net(xt, yt, xe, ye,
                       hidden_layers = [36,36,36],
                       initial_params = {'weight':0, 'bias':0},
                       criterion=nn.BCELoss, optimizer=torch.optim.SGD,
                       activation='sigmoid',
                       epochs=500, batch_size=30,
                       lr=0.1, momentum=0,
):

    train_data = Load_Data(xt.values, yt.to_frame().values)
    test_data = Load_Data(xe.values, ye.to_frame().values)
    
    ### ------------------------- Model Settings -------------------------------
    torch.manual_seed(1)
    Layers = Build_Layers(train_data, hidden_layers)
    model = Deep_NN(Layers)

    ### ------------------------------ Criterion, Optimizer and Train Loader ------------------------------
    criterion = criterion()
    optimizer = optimizer(model.parameters(), lr=lr, momentum=momentum)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    
    ###  ----------------------------------------- Training Model Parameters -------------------------
    LOSS, ACC, test_ACC = train(
        train_data=train_data, model=model,
        criterion=criterion, train_loader=train_loader, optimizer=optimizer,
        epochs=epochs, test_data=test_data,
    )
    
    return (LOSS, ACC, test_ACC)

def plot_learning_curve_comparison(eval_i, figsize=(15,3), tight_layout=True, legend=True):
    
    fig_ax = (fig, ax) = plt.subplots(1, 3, figsize=figsize, tight_layout=tight_layout)
    for label, (LOSS, ACC, TEST_ACC) in eval_i.items():
        plot_learning_curve(LOSS, ACC, TEST_ACC, figsize, tight_layout, fig_ax=fig_ax, label=label)
    if legend:
        for axi in ax: axi.legend()
    plt.show()

def plot_learning_curve(LOSS, ACC, TEST_ACC, figsize=(15,3), tight_layout=True, fig_ax=None, label=None):
    
    if fig_ax is None: fig, ax = plt.subplots(1, 3, figsize=figsize, tight_layout=tight_layout)
    else: fig, ax = fig_ax
    ax[0].plot(LOSS, label=label); ax[1].plot(ACC, label=label); ax[2].plot(TEST_ACC, label=label)
    ax[0].set(title='Loss', xlabel='epochs', ylabel='loss')
    ax[1].set(title='Train Accuracy', xlabel='epochs', ylabel='accuracy')
    ax[2].set(title='Test Accuracy', xlabel='epochs', ylabel='accuracy')
    if fig_ax is None: plt.show()