import torch
import torch.nn as nn
import torch.nn.functional as F

# Proposed Neural Network for Open Data BCN
class topological_NN(nn.Module):
    def __init__(self):
        super(topological_NN, self).__init__()

        # Convolutional layers for neighborhoods
        self.layer1_n = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        
        self.layer2_n = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Convolutional layers for districts + neighborhoods
        self.conv1 = nn.Conv2d(16 + 1, 32, kernel_size=5)
        self.bn = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1)      
        self.bn3 = nn.BatchNorm2d(64)
        
        # Feature classification with fully connected layers
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, 1)   

    def forward(self, district_map, neighborhood_map):

        # Evaluate the neighborhood adjacency matric
        y1 = self.layer1_n(neighborhood_map)
        y1 = self.layer2_n(y1)

        # Concatenate with the district adjaceny matris
        y = torch.cat((y1, district_map), axis=1)

        # Standard convolutional neural network with contextual input
        y = F.relu(self.conv1(y))
        y = self.bn(y)
        y = F.relu(self.conv2(y))
        y = self.bn2(y)
        y = F.relu(self.conv3(y))
        y = self.bn3(y)
        y = F.relu(self.fc1(y.view(y.size(0), -1)))
        y = F.relu(self.fc2(y))
        pred = self.fc3(y)

        return pred