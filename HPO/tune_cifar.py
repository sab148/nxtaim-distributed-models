# load general modules
import argparse
import os
import time
import numpy as np

# load torch and torchvision modules 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
from torchvision import datasets, transforms, models

# load ray modules
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler
from ray.air import session, RunConfig
from ray.train import Checkpoint

import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.tune.tuner import Tuner, TuneConfig

def parsIni():
    parser = argparse.ArgumentParser(description='Ray Tune Cifar-10 Example')
    parser.add_argument('--num-samples', type=int, default=24, metavar='N',
                    help='number of samples to train (default: 24)')
    parser.add_argument('--max-iterations', type=int, default=10, metavar='N',
                    help='maximum iterations to train (default: 10)')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                    help='parallel workers to train on a single trial (default: 1)')
    parser.add_argument('--scheduler', type=str, default='RAND',
                    help='scheduler for tuning (default: RandomSearch)')
    parser.add_argument('--data-dir', type=str, default='',
                    help='data directory for cifar-10 dataset')
    return parser

def accuracy(output, target):
    """! function that computes the accuracy of an output and target vector 
    @param output vector that the model predicted
    @param target actual  vector
    
    @return correct number of correct predictions
    @return total number of total elements
    """
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    
    # count correct classifications
    correct = pred.eq(target.view_as(pred)).cpu().float().sum()
    
    # count total samples
    total = target.size(0)
    return correct, total

def par_sum(field):
    """! function that sums a field across all workers to a worker
    @param field field in worker that should be summed up
    
    @return sum of all fields
    """
    # convert field to tensor
    res = torch.Tensor([field])
    
    # move field to GPU/worker
    res = res.cuda()
    
    # AllReduce operation
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    
    return res

def load_data(data_dir=None):
    """! function that loads training and validation set of cifar-10
    @param data_dir directory where the data is stored
    
    @return train_set training set of cifar-10
    @return val_set set of cifar-10
    """
    
    # vision preprocessing values
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    # transformations for the training set 
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=data_transforms
    )

    # Define the split sizes for training and validation sets
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size

    # Split the dataset randomly
    train_subset, val_subset = torch.utils.data.random_split(train_set, [train_size, val_size])

    return train_subset, val_subset
    
def train_cifar(config):
    """! function to train a ResNet on cifar-10 with different hyperparameters
    @param config hyperparameter search space
    """    

    # load a ResNet model
    model = models.resnet18()
    
    # prepare the model for Ray Tune
    model = train.torch.prepare_model(model)
        
    # define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"]*dist.get_world_size())

    # load the training and validation data
    train_set, val_set = load_data(str(config["data_dir"]))
    
    current_epoch = 0
    
    # Load existing model and optimizer checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            # Load checkpoint
            checkpoint = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))

            # Restore epoch value, model state, and optimizer state
            current_epoch = checkpoint['current_epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # define the train and validation dataloader
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=256,
        shuffle=True,
        num_workers=10)
    
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=256,
        shuffle=False,
        num_workers=10)

    # prepare the dataloaders for Ray Tune
    train_loader = train.torch.prepare_data_loader(train_loader)
    val_loader = train.torch.prepare_data_loader(val_loader)
        
    # prepare metrics
    train_acc = 0
    train_correct = 0
    train_total = 0
    
    val_acc = 0
    val_correct = 0
    val_total = 0
    
    # training and validation loop
    for epoch in range(current_epoch, config["max_iterations"]):
        # prepare model for training and loop over training dataset
        model.train()
        for i, (images, target) in enumerate(train_loader):

            # compute output
            optimizer.zero_grad()
            output = model(images)

            # compute loss
            loss = criterion(output, target)

            # count correct classifications
            tmp_correct, tmp_total = accuracy(output, target)    
            train_correct +=tmp_correct
            train_total +=tmp_total

            # backpropagation and optimization step
            loss.backward() 
            optimizer.step()

        # average the train metrics over all workers
        train_correct = par_sum(train_correct)
        train_total = par_sum(train_total)

        # compute final training accuracy
        train_acc = train_correct/train_total
        
        # prepare model for validation and loop over validation dataset
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader): 

                # compute output
                output = model(images)

                # count correct classifications
                tmp_correct, tmp_total = accuracy(output, target)
                val_correct +=tmp_correct
                val_total +=tmp_total    

            # average the validation metrics over all workers
            val_correct = par_sum(val_correct)
            val_total = par_sum(val_total)

            # compute final validation accuracy
            val_acc = val_correct/val_total
         
        # Save current state of model and optimizer
        os.makedirs("tune_model", exist_ok=True)
        torch.save({
            'current_epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, "tune_model/checkpoint.pt")

        checkpoint = Checkpoint.from_directory("tune_model")

        # report the training and validation accuracy back to the head node of Ray Tune
        session.report({"train_acc": train_acc.item(), "val_acc": val_acc.item()}, checkpoint = checkpoint)

def main(args):
    """! main function
    @param args input arguments
    """    
    
    # initalize Ray with the correct adress and node ip adress
    ray.init(address=os.environ['ip_head'], _node_ip_address=os.environ["head_node_ip"])  
    
    # define the hyperparameter search space 
    config = {
        # actual hyperparameters
        "lr": tune.loguniform(10e-5, 1),
        ## TODO ##
        # add more hyperparameters here
        
        # general configuration parameters 
        "data_dir": tune.choice([args.data_dir]),
        "max_iterations": tune.choice([args.max_iterations])
    }
    
    if (args.scheduler == "RAND"):
        # random scheduler
        scheduler = None
        search_alg = None
    
    # select a hyperparameter optimization algorithm
    if (args.scheduler == "ASHA"):
        # Asynchronous Successive Halving Algorithm
        scheduler = ASHAScheduler(
           # the number of iterations to allow the trials to run at max 
           max_t=args.max_iterations,
           # how many inital iterations before a bad trials get terminated 
           grace_period=1,
           # which percentage of trials to terminate
           reduction_factor=2)
        
        # set search algorithm
        search_alg = None
      
    if (args.scheduler == "HB"):
        scheduler = HyperBandScheduler(
            # the number of iterations to allow the trials to run at max
            max_t=args.max_iterations,
            # which percentage of trials to terminate
            reduction_factor=2)
        search_alg = None

    
    # define the general RunConfig of Ray Tune
    run_config = RunConfig(
        # name of the training run (directory name).
        name="cifar_tune",
        # directory to store the ray tune results in .
        storage_path=os.path.join(os.path.abspath(os.getcwd()), "ray_results"),
        # logger
        # stopping criterion when to end the optimization process
        stop={"training_iteration": args.max_iterations}
    )
    
    # wrapping the torch training function inside a TorchTrainer logic
    trainer = TorchTrainer(
        # torch training function
        train_loop_per_worker=train_cifar,
        # setting the default resources/workers to use for the training function, including the number of CPUs and GPUs
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=True, resources_per_worker={"CPU": 10, "GPU": 1}),
    )
    
    # defining the hyperparameter tuner 
    tuner = Tuner(
        # function to tune
        trainer,
        # hyperparameter search space
        param_space={"train_loop_config": config},
        # the tuning configuration
        tune_config=TuneConfig(
           # define how many trials to evaluate 
           num_samples=args.num_samples,
           # define which metric to use for measuring the performance of the trials
           metric="val_acc",
           # if the metric should be maximized or minimized 
           mode="max",
           # define which scheduler to use 
           scheduler=scheduler,
            # define which search algorithm to use
           search_alg=search_alg,
           ),
        run_config=run_config
    )
    
    # measure the total runtime
    start_time = time.time()
    
    # start the optimization process
    result = tuner.fit()
    
    runtime = time.time() - start_time
    
    # print total runtime
    print("Total runtime: ", runtime)

    # print metrics of the best trial
    best_result = result.get_best_result(metric="val_acc", mode="max")    
    
    print("Best result metrics: ", best_result) 

    # print results dataframe
    print("Result dataframe: ")
    print(result.get_dataframe().sort_values("val_acc", ascending=False))

if __name__ == "__main__":
    # get custom arguments from parser
    parser = parsIni()
    args = parser.parse_args()
    
    # call the main function to launch Ray
    main(args)

# eof