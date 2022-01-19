from collections import OrderedDict
import numpy as np 
import os
from typing import List, Tuple, Dict

import torch 
from argparse import ArgumentParser 

import src.py.flwr as fl 
import utils_mp as utils
from utils import seed_everything  

import multiprocessing as mp

import wandb 

import warnings

warnings.filterwarnings("ignore")
seed = 1234
seed_everything(seed)

# Setting up GPU for processing or CPU if GPU isn't available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Client(fl.client.NumPyClient):
    """Flower client implementing melanoma classification using PyTorch."""

    def __init__(
        self, 
    ) -> None:
        self.parameters = None 

    def get_parameters(self) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return self.parameters

    def get_properties(self, config):
        return {}

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.parameters = parameters

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set parameters from the global model
        self.set_parameters(parameters)
        # Prepare multiprocess
        manager = mp.Manager()
        # We receive the results through a shared dictionary
        return_dict = manager.dict()
        # Create the process
        p = mp.Process(target=utils.train, args=(args.model, parameters, return_dict, args.partition, 
                                                    args.num_partitions, args.log_interval, 1))
        # Start the process
        p.start() 
        # Wait for it to end
        p.join()
        # Close it
        try:
            p.close()
        except ValueError as e:
            print(f"Coudln't close the training process: {e}")
        # Get the return values
        new_parameters = return_dict["parameters"]
        data_size = return_dict["data_size"]
        # Del everything related to multiprocessing
        del (manager, return_dict, p)
        return new_parameters, data_size, {return_dict["train_loss"], return_dict["train_acc"], return_dict["val_loss"], return_dict["val_acc"]} 

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
    parser.add_argument("--model", type=str, default='efficientnet') 
    parser.add_argument("--log_interval", type=int, default='100')  
    parser.add_argument("--num_partitions", type=int, default='10') 
    parser.add_argument("--partition", type=int, default='0')  
    args = parser.parse_args()

    wandb.init(project="dai-healthcare" , entity='eyeforai', config={"model": args.model})

    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn")

    """ 
    # Load model
    model = utils.load_model(args.model)

    # Load data
    trainset, testset, num_examples = utils.load_isic_data()
    trainset, testset, num_examples = utils.load_partition(trainset, testset, num_examples, idx=args.partition, num_partitions=args.num_partitions)
    train_loader = DataLoader(trainset, batch_size=32, num_workers=4, shuffle=True) 
    test_loader = DataLoader(testset, batch_size=16, shuffle = False)   
    """
    
    # Start client 
    fl.client.start_numpy_client("0.0.0.0:8080", Client())

    