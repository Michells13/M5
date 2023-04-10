from torch import nn
from torchvision import models
import copy
import torchvision
import torch


class ResNet_Triplet_COCO(nn.Module):
    
    def __init__(self):
        """
        Loads the pretrained Resnet50 in COCO (from pretrained FasterRCNN)

        Returns
        -------
        None.

        """
        super().__init__()
        
        # Create the model
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        resnet = model.backbone.body
        
        # Check for all FrozenBN layers
        bn_to_replace = []
        for name, module in resnet.named_modules():
            if isinstance(module, torchvision.ops.misc.FrozenBatchNorm2d):
                #print('adding ', name)
                bn_to_replace.append(name)
        
        
        # Iterate all layers to change
        for layer_name in bn_to_replace:
            # Check if name is nested
            *parent, child = layer_name.split('.')
            # Nested
            if len(parent) > 0:
                # Get parent modules
                m = resnet.__getattr__(parent[0])
                for p in parent[1:]:    
                    m = m.__getattr__(p)
                # Get the FrozenBN layer
                orig_layer = m.__getattr__(child)
            else:
                m = resnet.__getattr__(child)
                orig_layer = copy.deepcopy(m) # deepcopy, otherwise you'll get an infinite recusrsion
            # Add your layer here
            in_channels = orig_layer.weight.shape[0]
            bn = nn.BatchNorm2d(in_channels)
            with torch.no_grad():
                bn.weight = nn.Parameter(orig_layer.weight)
                bn.bias = nn.Parameter(orig_layer.bias)
                bn.running_mean = orig_layer.running_mean
                bn.running_var = orig_layer.running_var
            m.__setattr__(child, bn)
    
    
        # Fix the bn1 module to load the state_dict
        resnet.bn1 = resnet.bn1.bn1
        
        # Create reference model and load state_dict
        reference = models.resnet50()
        reference.load_state_dict(resnet.state_dict(), strict=False)
        
        fc = nn.Identity()
        
        reference.fc = fc
        
        self.model = reference
        
    def forward(self,x):
        features = self.model(x)
        
        return features