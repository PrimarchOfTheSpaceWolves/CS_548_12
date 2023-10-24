import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torchvision.models import (list_models, 
                                get_model, 
                                get_weight, 
                                get_model_weights)
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torchvision.io import read_image

#print(list_models())

class CatDogDataset(Dataset):
    def __init__(self, base_dir, is_training,
                 transform=None, target_transform=None):
        super().__init__()
        all_filenames = os.listdir(base_dir)
        self.base_dir = base_dir
        
        train_list, test_list = train_test_split(
                                    all_filenames,
                                    train_size=0.70,
                                    random_state=42)
                
        if is_training:
            self.filenames = train_list
        else:
            self.filenames = test_list
            
        self.transform = transform
        self.target_transform = target_transform
                
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filepath = os.path.join(self.base_dir, 
                                self.filenames[idx])
        image = read_image(filepath)
        
        if "dog" in self.filenames[idx]:
            label = 0
        else:
            label = 1
            
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label
    

class NeuralNetwork(nn.Module):
    def __init__(self, class_cnt, input_channels):
        super().__init__()
        self.class_cnt = class_cnt
        self.input_channels = input_channels
        self.net_stack = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(1600, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, class_cnt)           
        )
        
    def forward(self, x):
        logits = self.net_stack(x)
        return logits

def main():
    writer = SummaryWriter(log_dir="./pylogs")
    
    data_transform = v2.Compose([
        v2.Resize((32,32)),
        v2.ToImageTensor(),
        v2.ConvertImageDtype()
    ])
    
    aug_transform = v2.Compose([
        v2.Resize((32,32)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImageTensor(),
        v2.ConvertImageDtype()
    ])
    
    '''
    class_cnt = 10
    input_channels = 3
    training_dataset = datasets.CIFAR10("data", train=True,
                                     download=True,
                                     transform=aug_transform)
    
    base_training_dataset = datasets.CIFAR10("data", train=True,
                                     download=True,
                                     transform=data_transform)
    
    testing_dataset = datasets.CIFAR10("data", train=False,
                                    download=True,
                                    transform=data_transform)
    '''
    
    class_cnt = 2
    input_channels = 3
    training_dataset = CatDogDataset(is_training=True,  
                                     base_dir="../catdog",                                   
                                     transform=aug_transform)
    
    base_training_dataset = CatDogDataset(is_training=True, 
                                          base_dir="../catdog",                                     
                                     transform=data_transform)
    
    testing_dataset = CatDogDataset(is_training=False, 
                                     base_dir="../catdog",                                    
                                    transform=data_transform)
    
    batch_size = 64
    train_dataloader = DataLoader(training_dataset,
                                  batch_size=batch_size)
    base_train_dataloader = DataLoader(base_training_dataset,
                                  batch_size=batch_size)
    test_dataloader = DataLoader(testing_dataset,
                                 batch_size=batch_size)
    
    model = NeuralNetwork(class_cnt, input_channels)
    
    '''
    model = get_model("vgg19", weights="DEFAULT")
    weights = get_weight("VGG19_Weights.DEFAULT")
    preprocess = weights.transforms()
    
    for param in model.parameters():
        param.requires_grad = False
    
    feature_cnt = model.classifier[0].in_features
    
    model.classifier = nn.Sequential(
        nn.Linear(feature_cnt, 32),
        nn.ReLU(),
        nn.Linear(32, class_cnt))
    '''
        
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    model = model.to(device)
    print("Using device", device)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    total_epochs = 10
    
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X,y) in enumerate(dataloader):
            #X = preprocess(X)
            
            X = X.to(device)
            y = y.to(device)
            
            pred = model(X)
            loss = loss_fn(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 100 == 0:
                loss = loss.item()
                current = (batch+1)*len(X)
                print("Loss:", loss, 
                      " at", current, 
                      "of", size)
                
    def test(dataloader, model, loss_fn, data_name):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for X,y in dataloader:
                #X = preprocess(X)
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(data_name, "EVAL: Loss:", test_loss,
              " Acc:", (correct*100))
        return test_loss
                
    checkpoint_freq = 2
    
    checkpoint_filename = "checkpoint.pt"
    start_epoch = 0
    if os.path.exists(checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename)
        start_epoch = checkpoint["epoch"]+1
        model.load_state_dict(checkpoint["network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loading previous checkpoint...")
                            
    for epoch in range(start_epoch, total_epochs):
        print("** EPOCH", (epoch+1), "***********")
        train(train_dataloader, model, loss_fn, optimizer)
        
        train_loss = test(base_train_dataloader, model,
                          loss_fn, "TRAIN")
        test_loss = test(test_dataloader, model, loss_fn,
                         "TEST")
        
        writer.add_scalars("Loss",
                          {
                             "Train": train_loss,
                             "Test": test_loss 
                          }, epoch)
        
        if epoch % checkpoint_freq == 0:
            save_info = {
                "epoch": epoch,
                "network": model.state_dict(),
                "optimizer": optimizer.state_dict()                
            }
            torch.save(save_info, checkpoint_filename)
            
    save_info = {
                "epoch": total_epochs,
                "network": model.state_dict(),
                "optimizer": optimizer.state_dict()                
                }
    torch.save(save_info, "final_model.pt")
        
    writer.flush()
    writer.close()
    
        
            

if __name__ == "__main__":
    main()
    
