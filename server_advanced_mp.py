#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : server_advanced_mp.py
# Modified   : 17.02.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

import sys 
import src.py.flwr as fl 
from typing import List, Tuple, Dict, Optional 
import torch 
import utils
import warnings
import wandb
from argparse import ArgumentParser 
import multiprocessing as mp

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXCLUDE_LIST = [
    #"num_batches_tracked",
    #"running",
    #"bn", #FedBN
]
seed = 2022
utils.seed_everything(seed)


def get_eval_fn(path):
    """Return an evaluation function for server-side evaluation.""" 

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Prepare multiprocess
        manager = mp.Manager()
        # We receive the results through a shared dictionary
        return_dict = manager.dict()
        # Create the process
        p = mp.Process(target=utils.val_mp_server, args=(args.model, weights, EXCLUDE_LIST, return_dict, device, path))
        # Start the process
        p.start()
        # Wait for it to end
        p.join()
        # Close it
        try:
            p.close()
        except ValueError as e:
            print(f"Coudln't close the evaluating process: {e}")
        # Get the return values
        loss = return_dict["loss"]
        accuracy = return_dict["accuracy"]
        auc = return_dict["auc_score"]
        # Del everything related to multiprocessing
        del (manager, return_dict, p)

        wandb.log({'Server/loss': loss, "Server/accuracy": float(accuracy), "Server/auc": float(auc)})
        
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
    fed_eval = 1 
    return {"val_steps": val_steps, "fed_eval": fed_eval}



if __name__ == "__main__":

    parser = ArgumentParser()  
    parser.add_argument("--model", type=str, default='efficientnet-b2')
    parser.add_argument("--tags", type=str, default='Exp 5. FedAvg') 
    parser.add_argument("--path", type=str, default='/workspace/melanoma_isic_dataset') 
    parser.add_argument(
        "--r", type=int, default=10, help="Number of rounds for the federated training"
    )
    parser.add_argument(
        "--fc",
        type=int,
        default=5,
        help="Min fit clients, min number of clients to be sampled next round",
    )
    parser.add_argument(
        "--ac",
        type=int,
        default=5,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )
    args = parser.parse_args()
    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)

    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn") 

    # Load model for
        # 1. server-side parameter initialization
        # 2. server-side parameter evaluation
    model = utils.load_model(args.model, device).eval() 

    wandb.init(project="dai-healthcare" , entity='eyeforai', group='mp', tags=[args.tags], config={"model": args.model})
    wandb.config.update(args)
    
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit = fc/ac,
        fraction_eval = 1,
        min_fit_clients = fc,
        min_eval_clients = 2,
        min_available_clients = ac,
        eval_fn=get_eval_fn(args.path),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters= fl.common.weights_to_parameters(utils.get_parameters(model, EXCLUDE_LIST)),  
    )
    # del the net as we don't need it anymore
    del model

    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": rounds}, strategy=strategy)