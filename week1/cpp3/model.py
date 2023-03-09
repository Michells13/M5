import torch
import torch.nn as nn

class ccp3(nn.Module):
    def __init__(self):
        super(ccp3, self).__init__()
        
        self.conv11 = nn.Conv2d(3, 32, 3)
        self.conv12 = nn.Conv2d(32, 32, 3)
        self.conv21 = nn.Conv2d(32, 64, 3)
        self.conv22 = nn.Conv2d(64, 64, 3)
        self.conv31 = nn.Conv2d(64, 128, 3)
        self.conv32 = nn.Conv2d(128, 128, 3)
        
        self.fc = nn.Linear(128, 8)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):

        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv21(x)
        x = self.relu(x)
        x = self.conv22(x)
        x = self.relu(x)
        x = self.pool(x)
    
        x = self.conv31(x)
        x = self.relu(x)
        x = self.conv32(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # x shape is (B, 128, H, W,)
        x = torch.amax(x, (-1, -2))
        # x shape is (B, 128)
        x = self.fc(x)
        
        return x
    
    
if __name__ == "__main__":
    
    device = "cuda"
    model = ccp3().to(device)
    input = torch.rand((16, 3, 256, 256), device=device)
    print(model(input).shape)