from collections import OrderedDict
import numpy as np 
import os 

import cv2
from PIL import Image
import torch

from pathlib import Path 
import torch.nn as nn 
from torch import optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader 
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50 
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import wandb

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

training_transforms = transforms.Compose([#Microscope(),
                                        #AdvancedHairAugmentation(),
                                        transforms.RandomRotation(30),
                                        #transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        #transforms.ColorJitter(brightness=32. / 255.,saturation=0.5,hue=0.01),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) 

testing_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# Creating seeds to make results reproducible
def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 1234
seed_everything(seed)


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
    
def set_weights(model, weights) -> None:
    # Set model parameters from a list of NumPy ndarrays
    keys = [k for k in model.state_dict().keys()]
    params_dict = zip(keys, weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True) 


class Net(nn.Module):
    def __init__(self, arch, return_feats=False):
        super(Net, self).__init__()
        self.arch = arch
        self.return_feats = return_feats
        if 'fgdf' in str(arch.__class__):
            self.arch.fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        if 'EfficientNet' in str(arch.__class__):   
            self.arch._fc = nn.Linear(in_features=1408, out_features=500, bias=True)
            #self.dropout1 = nn.Dropout(0.2)
        if 'resnet' in str(arch.__class__):   
            self.arch.fc = nn.Linear(in_features=2048, out_features=500, bias=True)
            
        self.output = nn.Linear(500, 1)
        
    def forward(self, images):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x = images
        features = self.arch(x)
        output = self.output(features)
        if self.return_feats:
            return features
        return output


def load_model(model = 'efficientnet'):
    arch = EfficientNet.from_pretrained('efficientnet-b2') if model=='efficientnet' else resnet50(pretrained=True)
    model = Net(arch=arch).to(DEVICE)

    return model


def create_split(source_dir, n_b, n_m):     
    # Split synthetic dataset  
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if os.path.isfile(f)]
    
    ind_0, ind_1 = [], []
    for i, f in enumerate(input_images):
        if f.split('.')[0][-1] == '0':
            ind_0.append(i)
        else:
            ind_1.append(i)  
    
    train_id_list, val_id_list  = ind_0[:round(len(ind_0)*0.8)],  ind_0[round(len(ind_0)*0.8):]       #ind_0[round(len(ind_0)*0.6):round(len(ind_0)*0.8)] ,
    train_id_1, val_id_1 = ind_1[:round(len(ind_1)*0.8)],  ind_1[round(len(ind_1)*0.8):] #ind_1[round(len(ind_1)*0.6):round(len(ind_1)*0.8)] ,
    
    train_id_list = np.append(train_id_list, train_id_1)
    val_id_list =   np.append(val_id_list, val_id_1)    
    
    return train_id_list, val_id_list  #test_id_list


def load_isic_data():
    # ISIC Dataset

    df = pd.read_csv('/workspace/melanoma_isic_dataset/train_concat.csv')
    train_img_dir = '/workspace/melanoma_isic_dataset/train/train/'
    
    df['image_name'] = [os.path.join(train_img_dir, df.iloc[index]['image_name'] + '.jpg') for index in range(len(df))]

    train_split, valid_split = train_test_split (df, stratify=df.target, test_size = 0.20, random_state=42) 
    train_df=pd.DataFrame(train_split)
    validation_df=pd.DataFrame(valid_split) 
    
    training_dataset = CustomDataset(df = train_df, train = True, transforms = training_transforms ) 
    testing_dataset = CustomDataset(df = validation_df, train = True, transforms = testing_transforms ) 

    num_examples = {"trainset" : len(training_dataset), "testset" : len(testing_dataset)} 
    
    return training_dataset, testing_dataset, num_examples


def load_synthetic_data(data_path, n_imgs):
    # Synthetic Dataset
    input_images = [str(f) for f in sorted(Path(data_path).rglob('*')) if os.path.isfile(f)]
    y = [0 if f.split('.jpg')[0][-1] == '0' else 1 for f in input_images]
    
    n_b, n_m = [int(i) for i in n_imgs.split(',') ] 
    train_id_list, val_id_list = create_split(data_path, n_b , n_m) 
    train_img = [input_images[int(i)] for i in train_id_list]
    train_gt = [y[int(i)] for i in train_id_list]
    test_img = [input_images[int(i)] for i in val_id_list]
    test_gt = [y[int(i)] for i in val_id_list]
    #train_img, test_img, train_gt, test_gt = train_test_split(input_images, y, stratify=y, test_size=0.2, random_state=3)
    synt_train_df = pd.DataFrame({'image_name': train_img, 'target': train_gt})
    synt_test_df = pd.DataFrame({'image_name': test_img, 'target': test_gt})
    
    training_dataset = CustomDataset(df = synt_train_df, train = True, transforms = training_transforms ) 
    testing_dataset = CustomDataset(df = synt_test_df, train = True, transforms = testing_transforms ) 
    
    num_examples = {"trainset" : len(training_dataset), "testset" : len(testing_dataset)} 

    return training_dataset, testing_dataset, num_examples


def load_partition(trainset, testset, num_examples, idx, num_partitions = 5):
    """Load 1/5th of the training and test data to simulate a partition."""
    assert idx in range(num_partitions) 
    n_train = int(num_examples["trainset"] / num_partitions)
    n_test = int(num_examples["testset"] / num_partitions)

    train_partition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_partition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )

    num_examples = {"trainset" : len(train_partition), "testset" : len(test_partition)} 

    return (train_partition, test_partition, num_examples)


class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = True, transforms= None):
        self.df = df
        self.transforms = transforms
        self.train = train
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        img_path = self.df.iloc[index]['image_name']
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        images =Image.open(img_path)

        if self.transforms:
            images = self.transforms(images)
            
        labels = self.df.iloc[index]['target']

        if self.train:
            #return images, labels
            return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
        
        else:
            #return (images)
            return img_path, torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
def val(model, validate_loader, criterion = nn.BCEWithLogitsLoss()):          
    model.eval()
    preds=[]            
    all_labels=[]
    criterion = nn.BCEWithLogitsLoss()
    # Turning off gradients for validation, saves memory and computations
    with torch.no_grad():
        
        val_loss = 0 
    
        for val_images, val_labels in validate_loader:
        
            val_images, val_labels = val_images.to(DEVICE), val_labels.to(DEVICE)
        
            val_output = model(val_images)
            val_loss += (criterion(val_output, val_labels.view(-1,1))).item() 
            val_pred = torch.sigmoid(val_output)
            
            preds.append(val_pred.cpu())
            all_labels.append(val_labels.cpu())
        pred=np.vstack(preds).ravel()
        pred2 = torch.tensor(pred)
        val_gt = np.concatenate(all_labels)
        val_gt2 = torch.tensor(val_gt)
            
        val_accuracy = accuracy_score(val_gt2, torch.round(pred2))
        val_auc_score = roc_auc_score(val_gt, pred)
        val_f1_score = f1_score(val_gt, np.round(pred))

        return val_loss/len(validate_loader), val_auc_score, val_accuracy, val_f1_score