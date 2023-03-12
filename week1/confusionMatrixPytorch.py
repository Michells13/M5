import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        # Initialize weights using Glorot Normalization
        nn.init.xavier_normal_(self.depthwise.weight)
        nn.init.xavier_normal_(self.pointwise.weight)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class VGG_SP_NOFC(nn.Module):
    def __init__(self, config):
        super(VGG_SP_NOFC, self).__init__()
        # Separable conv 1
        self.sep1 = SeparableConv(3, 16)
        # Batch norm 1
        self.batch1 = nn.BatchNorm2d(16)
        
        # Separable conv 2
        self.sep2 = SeparableConv(16, 32)
        # Batch norm 2
        self.batch2 = nn.BatchNorm2d(32)
        
        # Separable conv 3
        self.sep3 = SeparableConv(32, 64)
        # Batch norm 3
        self.batch3 = nn.BatchNorm2d(64)
        
        # Separable conv 4
        self.sep4 = SeparableConv(64, 128)
        # Batch norm 4
        self.batch4 = nn.BatchNorm2d(128)
        
        # Separable conv 5
        self.sep5 = SeparableConv(128, 256)
        # Batch norm 5
        self.batch5 = nn.BatchNorm2d(256)
        
        # FC
        self.fc = nn.Linear(256, config["n_class"])
        nn.init.xavier_normal_(self.fc.weight)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Global average pooling
        self.gAvPool = nn.AvgPool2d(kernel_size=16, stride=1)
        
        # Activations
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)
        
        

    def forward(self, x):
        # Sep 1
        x = self.pool(self.batch1(self.relu(self.sep1(x))))
        # Sep 2
        x = self.pool(self.batch2(self.relu(self.sep2(x))))
        # Sep 3
        x = self.pool(self.batch3(self.relu(self.sep3(x))))
        # Sep 4
        x = self.pool(self.batch4(self.relu(self.sep4(x))))
        # Sep 5
        x = self.batch5(self.relu(self.sep5(x)))
        # Global pooling
        x = self.gAvPool(x)
        x = torch.squeeze(x)
        # FC
        x = self.fc(x)
        
        # If it is in eval mode
        #if not self.training:
            # Softmax
        #    x = self.softmax(x)
        
        return x

modelPath = "model.pth"
train_data_dir = "./MIT_small_train_1/train"
test_data_dir = "./MIT_small_train_1/test"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

configs = dict(
    dataset = 'MIT_small_train_1',
    n_class = 8,
    image_width = 256,
    image_height = 256,
    batch_size = 32,
    model_name = 'VGG_SP_NOFC_keras',
    epochs = 100,
    init_learning_rate = 0.01,
    optimizer = 'nadam',
    loss_fn = 'categorical_crossentropy',
    metrics = ['accuracy'],
    weight_init = "glorot_normal",
    activation = "relu",
    regularizer = "l2",
    reg_coef = 0.01,
    # Data augmentation
    width_shift_range = 0,
    height_shift_range = 0,
    horizontal_flip = False,
    vertical_flip = False,
    rotation_range = 0,
    brightness_range = [0.8, 1.2],
    zoom_range = 0.15,
    shear_range = 0

)

transformNDA = transforms.Compose([
    transforms.Resize((configs["image_height"], configs["image_width"])),
    transforms.ToTensor(),
    #lambda x: x/255.
])

val_dataset = ImageFolder(test_data_dir, transform=transformNDA)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
labels = ["Opencountry", "coast", "forest", "highway", "inside_city", "mountain", "street", "tallbuilding"]


modelW = torch.load(modelPath)
model = VGG_SP_NOFC(config = configs)
model.load_state_dict(modelW["state_dict"])
model.eval()
model = model.to(device)

y_pred = np.zeros((0,8))
y_test = np.zeros((0))
with torch.no_grad():
    for X, y in val_loader:
        y_test = np.concatenate((y_test, y.numpy()))
        X = X.to(device)
        ypred = model(X)
        y_pred = np.vstack((y_pred, ypred.cpu().numpy()))

y_test_1 = y_test
y_pred_1 = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test_1, y_pred_1, normalize='true')
cm = np.around(cm, decimals = 2)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.show()

