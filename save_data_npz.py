#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : train_local.py
# Modified   : 17.02.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

from tkinter import image_names
import torch 
from torch.utils.data import DataLoader 
from argparse import ArgumentParser 

import utils  

import wandb 

import warnings

import os
import PIL.Image as Image
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms 

warnings.filterwarnings("ignore")
seed = 2022
utils.seed_everything(seed)

# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([   transforms.RandomRotation(30), 
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),  
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])]) 
test_transform = transforms.Compose([ 
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])]) 
def loadData(dataDir):
    # load data from npz format to numpy 
    path = os.path.join(dataDir)
    with np.load(path) as f:
        xTrain, yTrain, xTest, yTest = f['xtrain'], f['ytrain'] , f['xtest'], f['ytest'] 
        xTrain = xTrain.transpose(0,3,1,2) / 255.0  
        xTest = xTest.transpose(0,3,1,2) / 255.0   


    # transform numpy to torch.Tensor
    xTrain, yTrain, xTest, yTest = map(torch.tensor, (xTrain.astype(np.float32), 
                                                      yTrain.astype(np.float32), 
                                                      xTest.astype(np.float32),
                                                      yTest.astype(np.float32)))    

    xTrain = train_transform(xTrain)
    xTest = test_transform(xTest)
    # convert torch.Tensor to a dataset 
    trainDs = torch.utils.data.TensorDataset(xTrain,yTrain)
    testDs = torch.utils.data.TensorDataset(xTest,yTest)
    return trainDs, testDs


if __name__ == "__main__":
    parser = ArgumentParser()  
    parser.add_argument("--path_data", type=str, default='/workspace/melanoma_isic_dataset')  
    parser.add_argument("--num_partitions", type=int, default='10') 
    parser.add_argument("--partition", type=int, default='0')   
    args = parser.parse_args()
  
    trainDs, testDs = loadData("/workspace/flower/isic_partitionExp3_client0.npz") 
    train_loader = DataLoader(trainDs, batch_size=32, num_workers=8, worker_init_fn=utils.seed_worker ,shuffle=True) 
    test_loader = DataLoader(testDs, batch_size=16, num_workers=4, worker_init_fn=utils.seed_worker, shuffle = False)   
    model = utils.load_model("efficientnet-b2")
    model = utils.train(model, train_loader, test_loader, {"trainset" :len(trainDs), "testset": len(testDs)}, args.partition, True)

    # Save Data in npz for SL HPE

    #trainset, testset, num_examples = utils.load_isic_data(args.path_data)
    #trainset, testset, num_examples = utils.load_partition(trainset, testset, num_examples, idx=args.partition, num_partitions=args.num_partitions)

    # trainset, testset, num_examples = utils.load_exp1_partition(trainset, testset, num_examples, idx=args.partition)
    
    traindf, testsdf, num_examples = utils.load_isic_by_patient(args.partition) 

    imgs = []
    labels = []
    for row in range(len(traindf)):
        img_path = traindf.iloc[row]['image_name']
        label = traindf.iloc[row]['target']
        image = np.array(Image.open(img_path))
        imgs.append(image)
        labels.append(label)
    xtrain = np.array(imgs)
    ytrain = np.array(labels)
    imgs = []
    labels = []
    for row in range(len(testsdf)):
        img_path = testsdf.iloc[row]['image_name']
        label = testsdf.iloc[row]['target']
        image = np.array(Image.open(img_path))
        imgs.append(image)
        labels.append(label)
    xtest = np.array(imgs)
    ytest = np.array(labels)
    np.savez_compressed('/workspace/flower/isic_partitionExp3_client' + str(args.partition), xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest )  


    #
    