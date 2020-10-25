import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# import topology
import datasets
from torch.utils.data import Dataset, DataLoader
import torch


from model import topological_NN

def train_model(dataset, num_epochs, batch_size, lr=0.001, early_stopping=False):

    # Neural Network initialization
    model = topological_NN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_train_values = []
    loss_val_values = []

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [int(len(dataset)*0.7), len(dataset) - int(len(dataset)*0.7)])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True, num_workers=0)

    for epoch in range(num_epochs):
        
        loss_train = 0
        loss_val = 0

        for i_batch, sample_batched in enumerate(train_loader):
            output = model(sample_batched['district_graph'].unsqueeze(1).float(), sample_batched['neighborhood_graph'].unsqueeze(1).float())
            loss = criterion(output, sample_batched['label'].unsqueeze(1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train = loss_train + loss.item()

        for i_batch, sample_batched in enumerate(val_loader):
            output = model(sample_batched['district_graph'].unsqueeze(1).float(), sample_batched['neighborhood_graph'].unsqueeze(1).float())
            loss = criterion(output, sample_batched['label'].unsqueeze(1).float())
            loss_val = loss_val + loss.item()
        
        loss_train_values.append(loss_train)    
        loss_val_values.append(loss_val)

        if epoch % 10 is 0:
            print('Epoch [{0:>3d}/{1}] \t Training Loss: {2:.4f} \t Validation Loss: {3:.4f}'.format(epoch, num_epochs, loss_train, loss_val))

    return model, loss_train_values, loss_val_values


def evaluate_model(model):

    # Load the connections dataset
    BCN_dataset = datasets.BCNDataset()
    evaluations = []

    # Iterate for every neighborhood and get its district and neighborhood weighted graoh
    for data in BCN_dataset:
        d_graph = torch.tensor(data['district_graph']).unsqueeze(0).unsqueeze(0).float()
        n_graph = torch.tensor(data['neighborhood_graph']).unsqueeze(0).unsqueeze(0).float()

        output = model(d_graph, n_graph)

        evaluation = {  'district_name':        data['district_name'],
                        'neighborhood_name':    data['neighborhood_name'],
                        'prediction':           output.item() }
        
        evaluations.append(evaluation)

    return evaluations
