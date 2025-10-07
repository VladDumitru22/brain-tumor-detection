import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size=128):
        super().__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),                        

            nn.Conv2d(6,16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)          
        )

        self.flattened_size = 16 * (input_size // 4) * (input_size // 4) 

        self.fc_model = nn.Sequential(
            nn.Linear(self.flattened_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x
