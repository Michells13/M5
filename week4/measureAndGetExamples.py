from metrics import *
from torchvision.models import resnet18, ResNet18_Weights
from pytorch_metric_learning import testers
from similarRetrieval import FAISSretrieval, KNNretrieval
import torch
import torchvision.models as models
from torchvision import datasets
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

if __name__ == "__main__":
    
    # Device
    device = torch.device("cuda")
    weightsPath = "resnet_siamese_lr_1e-05_batchSize_128_miner_no.pth"
    
    # Init model
    model1 = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(*(list(model1.children())[:-1]))
    model = nn.Sequential(feature_extractor, nn.Flatten()).to(device)
    model.load_state_dict(torch.load(weightsPath, map_location=device))
    
    # Default transforms
    transformation_images = ResNet18_Weights.IMAGENET1K_V1.transforms()
    
    # Init datasets
    dataset_train = datasets.ImageFolder(root="./MIT_split/train/",transform=transformation_images)
    dataset_test = datasets.ImageFolder(root="./MIT_split/test/",transform=transformation_images)
    
    
    # Compute embeddings
    train_embeddings, train_labels = get_all_embeddings(dataset_train, model)
    test_embeddings, test_labels = get_all_embeddings(dataset_test, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    train_embeddings = train_embeddings.cpu().numpy()
    test_embeddings = test_embeddings.cpu().numpy()
    train_labels = train_labels.cpu().numpy()
    test_labels = test_labels.cpu().numpy()
    
    # Do retrieval
    retrieval = KNNretrieval(train_embeddings, "l2", train_embeddings.shape[0])
    (dis, neighbors) = retrieval.getMostSimilar(test_embeddings, train_embeddings.shape[0])
    
    results = []
    for i, label in enumerate(test_labels):
        results.append((train_labels[neighbors[i]] == label).astype(np.int32))
    results = np.array(results)
    
    # Metrics
    # Compute metrics
    print("P@1: ", mPrecisionK(results, 1))
    print("P@5: ", mPrecisionK(results, 5))
    print("MAP: ", MAP(results))
    
    classes = dataset_test.classes
    # Precision-Recall Curve
    for i, c in enumerate(classes):
        p, r = precisionRecall(results[test_labels==i])
        plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(classes)
    plt.show()

    # Re update dataset to visualize correclty
    dataset_train = datasets.ImageFolder(root="./MIT_split/train/")
    dataset_test = datasets.ImageFolder(root="./MIT_split/test/")

    inStr = input("Press Enter to continue, other key to exit...")
    while inStr == "":
        # Show results
        query = np.random.choice(list(range(test_embeddings.shape[0]))) #695
        print(query)
        print("Query image:")
        # Get image
        img, label = dataset_test[query]
        img = np.array(img)
        plt.imshow(img)
        plt.show()
        # Get label
        print("Label: ", classes[label])
        
        # Get 5 most close images
        for i in range(5):
            print(i, ". closest image:")
            
            neighbor = neighbors[query, i]
            
            # Get image
            img, labelTrain = dataset_train[neighbor]
            img = np.array(img)
            plt.imshow(img)
            plt.show()
            # Get values
            print("Label: ", classes[labelTrain])
        
        inStr = input("Press Enter to continue, other key to exit...")