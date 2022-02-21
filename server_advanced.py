#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : server_advanced.py
# Modified   : 17.02.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

import sys
sys.path.append('/workspace/flower')  

import src.py.flwr as fl 
from typing import List, Tuple, Dict, Optional 
import numpy as np
sys.path.append('/workspace/stylegan2-ada-pytorch') 
import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
from collections import OrderedDict
import utils
import warnings
import wandb
from argparse import ArgumentParser  

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EXCLUDE_LIST = [
    "num_batches_tracked",
    "running",
]
seed = 2022
utils.seed_everything(seed)

""" 
def set_parameters(model, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        keys = [k for k in model.state_dict().keys()] #  if 'bn' not in name]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
"""

def get_parameters(net) -> List[np.ndarray]:
        parameters = []
        for i, (name, tensor) in enumerate(net.state_dict().items()):
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


def set_parameters(net, parameters):
        keys = []
        for name in net.state_dict().keys():
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
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)

        
def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself

    # Exp 1
    trainset, testset, num_examples = utils.load_isic_data()
    trainset, testset, num_examples = utils.load_partition(trainset, testset, num_examples, idx=3, num_partitions=10)  # Use validation set partition 3 for evaluation of the whole model
    
    # Exp 2
    #_, testset, _ = utils.load_isic_by_patient_server()
    testloader = DataLoader(testset, batch_size=32, num_workers=4, worker_init_fn=utils.seed_worker, shuffle = False)  
    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters 
        set_parameters(model, weights) 
        loss, auc, accuracy, f1 = utils.val(model, testloader, criterion = nn.BCEWithLogitsLoss()) 
        """ 
        index_pos_list = [ i for i in range(len(keys)) if 'num_batches' in keys[i]]
        for i in index_pos_list:
            weights[i] = 96
        actual_weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
        OTRO: [ np.sum(np.abs(x - y)) for x, y in zip(weights, actual_weights) ]
        equal=np.array([(weights[i] == actual_weights[i]).all() for i in range(len(weights))])
        indexes = list(np.where(equal==False)[0])
        keys = [k for k in model.state_dict().keys()] 
        out = []
        for i in range(len(equal)):
            for o, a in (weights[i], actual_weights[i]):
                if equal[i]==False:
                    out.append([o,a]) """
        
        if not args.nowandb:
            wandb.log({'Server/loss': loss, "Server/accuracy": float(accuracy)})

        return float(loss), {"accuracy": float(accuracy), "auc": float(auc)}

    return evaluate
    

def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}



if __name__ == "__main__":

    parser = ArgumentParser()  
    parser.add_argument("--model", type=str, default='efficientnet-b2')
    parser.add_argument("--tags", type=str, default='Exp 2. FedAdagrad no BN') 
    parser.add_argument("--nowandb", action="store_true") 

    parser.add_argument(
        "-r", type=int, default=10, help="Number of rounds for the federated training"
    )
    parser.add_argument(
        "-fc",
        type=int,
        default=2,
        help="Min fit clients, min number of clients to be sampled next round",
    )
    parser.add_argument(
        "-ac",
        type=int,
        default=2,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )
    
    args = parser.parse_args()
    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)

    # Load model for
        # 1. server-side parameter initialization
        # 2. server-side parameter evaluation
    model = utils.load_model(args.model).eval()
    #model_weights = [val.cpu().numpy() for name, val in model.state_dict().items()] #  if 'bn' not in name] 

    if not args.nowandb:
        wandb.init(project="dai-healthcare" , entity='eyeforai', group='FL', tags=[args.tags] ,config={"model": args.model})
        wandb.config.update(args)
        # wandb.watch(model, log='all')
    
    # Create strategy
    strategy = fl.server.strategy.FedYogi(
        fraction_fit = fc/ac,
        fraction_eval = 0.2, # not used - no federated evaluation
        min_fit_clients = fc,
        min_eval_clients = 2, # not used 
        min_available_clients = ac,
        eval_fn=get_eval_fn(model),
        #on_fit_config_fn=fit_config,
        #on_evaluate_config_fn=evaluate_config,
        initial_parameters= fl.common.weights_to_parameters(get_parameters(model)),  
    )

    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": rounds}, strategy=strategy) 