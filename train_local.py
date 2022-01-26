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
    parser.add_argument("--tags", type=str, default='local_training') 
    parser.add_argument("--log_interval", type=int, default='100')  
    parser.add_argument("--epochs", type=int, default='15')  
    parser.add_argument("--early_stopping", type=int, default='3')  
    parser.add_argument("--num_partitions", type=int, default='10') 
    parser.add_argument("--partition", type=int, default='0', help="To train with the whole dataset use 'whole' ")  
    args = parser.parse_args()
    
    wandb.init(project="dai-healthcare" , entity='eyeforai', group='local_training', tags=[args.tags], config={"model": args.model})
    wandb.config.update(args) 

    # Load model
    model = utils.load_model(args.model)

    # Load data
    # trainset, testset, num_examples = utils.load_isic_data()
    # trainset, testset, num_examples = utils.load_partition(trainset, testset, num_examples, idx=args.partition, num_partitions=args.num_partitions)
    # trainset, testset, num_examples = utils.load_experiment_partition(trainset, testset, num_examples, idx=args.partition)
    
    if args.partition == 'whole':
        trainset, testset, num_examples = utils.load_isic_by_patient_server()
    else:
        trainset, testset, num_examples = utils.load_isic_by_patient_client(args.partition) 
    
    train_loader = DataLoader(trainset, batch_size=32, num_workers=4, shuffle=True) 
    test_loader = DataLoader(testset, batch_size=16, shuffle = False)   
    print(num_examples)
    
    utils.train(model, train_loader, test_loader, num_examples, args.partition,args.log_interval, epochs=args.epochs, es_patience=3)
        