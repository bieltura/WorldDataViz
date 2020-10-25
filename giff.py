# Script to train the network and generate the output giffs in the visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import imageio
import os

from train import evaluate_model    
from model import topological_NN
import datasets

#dataset = datasets.LloguersDataset(years=[2019,2018,2017], lloguer_mensual=True)
#dataset = datasets.AtursDataset(years=[2019, 2018], specific_months=[6,7,8])
#dataset = datasets.AtursDataset(years=[2019, 2018])
#dataset = datasets.EspVidaDataset()
#dataset = datasets.SuperficieHabitatge(years=[2019, 2018])
#dataset = datasets.SuperficieLloguer(years=[2019,2018,2017,2016])
dataset = datasets.EdatMitjana(years=[2019, 2018])

data_name = dataset.get_title()
print("Dataset Loaded: {0}".format(data_name))
skipped_epoch = 4

num_epochs = 150
batch_size = 200
lr = 0.0005

heatmaps_dir = "output/giff/heatmap/"
loss_dir = "output/giff/loss/"

def generate_img(epoch, model, loss_train, loss_val):

    # Visualization of the loss value evolution (discarding the firts epochs - random init)
    plt.plot(loss_train[skipped_epoch:], label="Train")
    plt.plot(loss_val[skipped_epoch:], label="Validation")

    plt.xlabel("Number of Epochs")
    plt.xlim(0, num_epochs)
    plt.ylabel("Loss Value")
    plt.legend(loc="upper right")
    plt.title("Training evaluation loss")

    # Save the figure
    plt.savefig(loss_dir + "{0:03d}.png".format(epoch))
    plt.cla()
    plt.clf()
    plt.close()

    evaluations = evaluate_model(model)
    predictions = []

    for evaluation in evaluations:
        predictions.append(evaluation['prediction'])

    # Load the district map GEOJSON and add the column predicition
    map_data = gpd.read_file("bcn_map/barris.geojson")
    map_data['PRED'] = predictions
    fig, ax = plt.subplots(figsize=(10, 10))
 
    # Axis names and titles 
    ax.set_title(data_name + " por barrio de Barcelona", fontdict={'fontsize': 15})
    plt.axis('off')

    map_data.plot(column='PRED', cmap='inferno', ax=ax, zorder=10, legend=True)
    plt.savefig(heatmaps_dir + "{0:03d}.png".format(epoch))
    plt.cla()
    plt.clf()
    plt.close()

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

    if epoch > skipped_epoch:
        generate_img(epoch, model, loss_train_values, loss_val_values)

    if epoch % 10 is 0:
        print('Epoch [{0:>3d}/{1}] \t Training Loss: {2:.4f} \t Validation Loss: {3:.4f}'.format(epoch, num_epochs, loss_train, loss_val))

heatmaps = os.listdir(heatmaps_dir)
loss = os.listdir(loss_dir)

print("Generating Heatmap giff...")
with imageio.get_writer('output/giff/heatmap.gif', mode='I', loop=1) as writer:
    for filename in heatmaps:
        image = imageio.imread(heatmaps_dir + filename)
        writer.append_data(image)
        os.remove(heatmaps_dir + filename)

print("Generating Loss evolution giff...")
with imageio.get_writer('output/giff/loss.gif', mode='I', loop=1) as writer:
    for filename in loss:
        image = imageio.imread(loss_dir + filename)
        writer.append_data(image)
        os.remove(loss_dir + filename)