#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : train_local.py
# Modified   : 17.02.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

from collections import OrderedDict
import numpy as np  
from typing import List, Tuple, Dict
import torch 
from torch.utils.data import DataLoader 
from argparse import ArgumentParser 
import src.py.flwr as fl 
import utils
from utils import Net, seed_everything  , training_transforms, testing_transforms


import wandb 

import warnings

warnings.filterwarnings("ignore")
seed = 1234
seed_everything(seed)

# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXCLUDE_LIST = [
    "running",
    "num_batches_tracked",
#"bn",
]

class Client(fl.client.NumPyClient):
    """Flower client implementing melanoma classification using PyTorch."""

    def __init__(
        self,
        model: Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_properties(self, config):
        return {}
    
    """
    def get_parameters(self) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for name, val in self.model.state_dict().items() ] # if 'bn' not in name]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        keys = [k for k in self.model.state_dict().keys() ] #if 'bn' not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)
    
    """
    def get_parameters(self) -> List[np.ndarray]:
        parameters = []
        for i, (name, tensor) in enumerate(self.model.state_dict().items()):
            # print(f"  [layer {i}] {name}, {type(tensor)}, {tensor.shape}, {tensor.dtype}")

            # Check if this tensor should be included or not
            exclude = False
            for forbidden_ending in EXCLUDE_LIST:
                if forbidden_ending in name:
                    exclude = True
            if exclude:
                continue

            # Convert torch.Tensor to NumPy.ndarray
            parameters.append(tensor.cpu().numpy())

        return parameters


    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        keys = []
        for name in self.model.state_dict().keys():
            # Check if this tensor should be included or not
            exclude = False
            for forbidden_ending in EXCLUDE_LIST:
                if forbidden_ending in name:
                    exclude = True
            if exclude:
                continue

            # Add to list of included keys
            keys.append(name)

        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)
    

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        self.model = utils.train(self.model, self.trainloader, self.testloader, self.num_examples, args.partition, 
                                args.nowandb, args.log_interval, epochs=args.epochs, es_patience=3)
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # WE DON'T EVALUATE OUR CLIENTS DECENTRALIZED
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, auc, accuracy, f1 = utils.val(self.model, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy), "auc": float(auc)}
                



if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--model", type=str, default='efficientnet-b2') 
    parser.add_argument("--log_interval", type=int, default='100')  
    parser.add_argument("--epochs", type=int, default='2')  
    parser.add_argument("--num_partitions", type=int, default='20') 
    parser.add_argument("--partition", type=int, default='0')   
    parser.add_argument("--tags", type=str, default='Exp 3. FedAdagrad') 
    parser.add_argument("--nowandb", action="store_true") 
    args = parser.parse_args()

    # Load model
    model = utils.load_model(args.model)

    if not args.nowandb:
        wandb.init(project="dai-healthcare" , entity='eyeforai', group='FL', tags=[args.tags], config={"model": args.model})
        wandb.config.update(args) 
        # wandb.watch(model, log="all")
    
    # Load data
    # Normal partition
    # trainset, testset, num_examples = utils.load_isic_data()
    # trainset, testset, num_examples = utils.load_partition(trainset, testset, num_examples, idx=args.partition, num_partitions=args.num_partitions)
    # Exp 1
    # trainset, testset, num_examples = utils.load_exp1_partition(trainset, testset, num_examples, idx=args.partition)
    # Exp 2/3
    train_df, validation_df, num_examples = utils.load_isic_by_patient(args.partition)
    trainset = utils.CustomDataset(df = train_df, train = True, transforms = training_transforms) 
    testset = utils.CustomDataset(df = validation_df, train = True, transforms = testing_transforms ) 
    train_loader = DataLoader(trainset, batch_size=32, num_workers=4, worker_init_fn=utils.seed_worker, shuffle=True) 
    test_loader = DataLoader(testset, batch_size=16, num_workers=4, worker_init_fn=utils.seed_worker, shuffle = False)   
    
    # Start client 
    client = Client(model, train_loader, test_loader, num_examples)
    fl.client.start_numpy_client("0.0.0.0:8080", client)

    