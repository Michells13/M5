from torch import nn
from torchvision import models
import copy
import torchvision
import torch
from torch.nn import Module, Linear, ReLU, init, Sequential, Dropout, LayerNorm



class ImgModel(Module):
    def __init__(self, dim=4096,embedding_size = 1000):
        super(ImgModel, self).__init__()

        self.linear1 = Linear(dim, embedding_size)
        self.activation = ReLU()
        self.init_weights()

    def init_weights(self):
        # Linear
        init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        x = self.activation(x)
        x = self.linear1(x)
        x = x / x.pow(2).sum(1, keepdim=True).sqrt()
        return x


class TextModel(Module):
    def __init__(self, embedding_size = 1000):
        super(TextModel, self).__init__()
        self.linear1 = Linear(300, embedding_size)
        self.activation = ReLU()

        self.init_weights()

    def init_weights(self):
        # Linear
        init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        x = self.activation(x)
        x = self.linear1(x)
        x = x / x.pow(2).sum(1, keepdim=True).sqrt()
        return x

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


class FasterRCNN_Triplet_COCO(nn.Module):
    
    def __init__(self, weighted = False):
        """
        Loads the pretrained Resnet50 in COCO (from pretrained FasterRCNN)

        Returns
        -------
        None.

        """
        super().__init__()
        
        # Mode
        self.weighted = weighted
        
        # Create the model
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size = 240, max_size = 320)
        
        self.features = []
        self.scores = []
        self.proposals = []
        def save_features(mod, inp, outp):
            self.features.append(outp)
        def save_scores(mod, inp, outp):
            self.scores.append(outp)
        def save_proposals(mod, inp, outp):
            self.proposals.append(inp)
        
        # you can also hook layers inside the roi_heads
        layerFeatures = 'roi_heads.box_head.fc7'
        layerClasses = "roi_heads.box_predictor.cls_score"
        layerProposals = "roi_heads.box_roi_pool"#­"rpn.head.cls_logits"
        for name, layer in model.named_modules():
            if name == layerFeatures:
                layer.register_forward_hook(save_features)
            
            if name == layerClasses:
                layer.register_forward_hook(save_scores)
                
            if name == layerProposals:
                layer.register_forward_hook(save_proposals)
    
    
        
        self.model = model
        
    def forward(self,x):
        
        target = {}
        target["boxes"] = torch.zeros((0,4)).to("cuda")
        target["labels"] = torch.zeros((0), dtype = torch.int64).to("cuda")
        target["image_id"] = torch.zeros((0), dtype = torch.int64).to("cuda")
        
        targets = [target]*x.shape[0]
        
        features = self.model(x, targets)
        
        
        # Obtain stored data
        proposalsScores = self.scores[0]
        del self.scores[0]
        proposals = self.proposals[0]
        del self.proposals[0]
        proposalsFeatures = self.features[0]
        del self.features[0]
        
        # Proposals per image        
        bbox_per_image = proposalsFeatures.shape[0]//x.shape[0]
        features_per_image = [bbox_per_image]*x.shape[0]
        
        # Split proposals into images
        proposalsFeatures = proposalsFeatures.split(features_per_image, 0)
        
        
        # Softmax per BBox proposal
        proposalsScores = torch.nn.functional.softmax(proposalsScores, dim = 1)
        # Get max value (confidence) per proposal
        proposalsScores = torch.max(proposalsScores, dim = 1)[0]
        # Split into images
        proposalsScores = proposalsScores.split(features_per_image, 0)
        # Obtain weights from confidences
        proposalsScores = [torch.nn.functional.softmax(a, dim = 0).unsqueeze(1) for a in proposalsScores]
        
        if self.weighted:
            # Use weighted sum of proposals
            features = [proposalsFeatures[i]*proposalsScores[i] for i in range(len(proposalsScores))]
            features = [torch.sum(f, dim = 0).unsqueeze(0) for f in features]
        else:
            # Use mean of all proposals
            features = proposalsFeatures
            features = [torch.mean(f, dim = 0).unsqueeze(0) for f in features]
        
        # Stack all images features
        features = torch.vstack(features)
        
        return features