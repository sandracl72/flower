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

warnings.filterwarnings("ignore")
seed = 2022
utils.seed_everything(seed)

# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--model", type=str, default='efficientnet-b2') 
    parser.add_argument("--path_data", type=str, default='/workspace/melanoma_isic_dataset') 
    parser.add_argument("--tags", type=str, default='local_training Exp 3', help="Use 'whole' for training with whole dataset") 
    parser.add_argument("--log_interval", type=int, default='100')  
    parser.add_argument("--epochs", type=int, default='20')  
    parser.add_argument("--early_stopping", type=int, default='3')  
    parser.add_argument("--num_partitions", type=int, default='10') 
    parser.add_argument("--partition", type=int, default='0')  
    parser.add_argument("--nowandb", action="store_true") 
    parser.add_argument("--test", action="store_true", help="evaluate with global testset") 
    args = parser.parse_args()
    
    if not args.nowandb:
        wandb.init(project="dai-healthcare" , entity='eyeforai', group='local_training', tags=[args.tags], config={"model": args.model})
        wandb.config.update(args) 

    # Load model
    model = utils.load_model(args.model)

    # Load data
    """ trainset, testset, num_examples = utils.load_isic_data(args.path_data)
    trainset, testset, num_examples = utils.load_partition(trainset, testset, num_examples, idx=args.partition, num_partitions=args.num_partitions)
    imgs = []
    for row in range(len(trainset.dataset.df)):
        img_path = trainset.dataset.df.iloc[row]['image_name']
        image = np.array(Image.open(img_path))
        imgs.append(image)
    imgs = np.array(imgs)
    np.savez_compressed('/workspace/isic_partition0', a=imgs)  """

    # trainset, testset, num_examples = utils.load_exp1_partition(trainset, testset, num_examples, idx=args.partition)
    
    if 'whole' in args.tags:
        trainset, testset, num_examples = utils.load_isic_by_patient_server()
    else:
        trainset, testset, num_examples = utils.load_isic_by_patient(args.partition) 
    
    train_loader = DataLoader(trainset, batch_size=32, num_workers=8, worker_init_fn=utils.seed_worker ,shuffle=True) 
    test_loader = DataLoader(testset, batch_size=16, num_workers=4, worker_init_fn=utils.seed_worker, shuffle = False)   
    print(num_examples)
        
    utils.train(model, train_loader, test_loader, num_examples, args.partition, args.nowandb,args.log_interval, epochs=args.epochs, es_patience=3)

    #Evaluate with global testset
    testset = utils.load_isic_by_patient_client(-1) 
    test_loader = DataLoader(testset, batch_size=16, num_workers=4, worker_init_fn=utils.seed_worker, shuffle = False)   
    val_loss, val_auc_score, val_accuracy, val_f1 = utils.val(model, test_loader)
    print( "Global testset: \n",
        "Validation Accuracy: {:.3f}".format(val_accuracy),
        "Validation AUC Score: {:.3f}".format(val_auc_score),
        "Validation F1 Score: {:.3f}".format(val_f1))

        