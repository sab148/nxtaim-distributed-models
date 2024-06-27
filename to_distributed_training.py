import argparse
import functools
import os
import time

import torch
from torch.distributed import checkpoint as dist_checkpoint
from torch.distributed import fsdp
import torchvision

from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import checkpoints_handler as model_checkpointing

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help=(
            'Step size or learning rate of the optimizer. '
            'Will be scaled according to the number of processes. '
            '(See `--batch-size`.)'
        ),
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help=(
            'How many samples to use per batch. '
            'Note that this is the local batch size; '
            'the effective, or global, batch size will be obtained by '
            'multiplying this number with the number of processes.'
        ),
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=120,
        help='How many epochs to train for.',
    )
    parser.add_argument(
        '--train-num-workers',
        type=int,
        default=0,
        help='How many workers to use for processing the training dataset.',
    )
    parser.add_argument(
        '--valid-num-workers',
        type=int,
        default=0,
        help='How many workers to use for processing the validation dataset.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random number generator initialization value.',
    )
    parser.add_argument(
        '--save-model-opt',
        choices=['full', 'sharded', 'local'],
        default='local',
        help='How to save the model state.',
    )
    parser.add_argument(
        '--save-optimizer',
        action='store_true',
        default=False,
        help='Whether to save the optimizer state.',
    )

    args = parser.parse_args()
    return args

## TODO:

# Add utility functions to return if the current process is rank 0.


# Add utility functions to print messages only from rank 0.


# Add utility functions to save the model.


def prepare_datasets(args, device):
    """Return the train, validation, and test datasets already wrapped
    in a dataloader.
    """
    dataset = torchvision.datasets.FakeData(
        transform=torchvision.transforms.ToTensor(),
    )

    valid_length = len(dataset) // 10
    test_length = len(dataset) // 20
    train_length = len(dataset) - valid_length - test_length
    train_dset, valid_dset, test_dset = torch.utils.data.random_split(
        dataset,
        [train_length, valid_length, test_length],
    )

    ## TODO:
    # 1. Create a DistributedSampler object for each set.
    # ** shuffle=True only for training set
    # 2. Pass the DistributedSampler object to the DataLoader object for each set.

    train_dset = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size,
        # Uses the distributed sampler to ensure each process gets a different subset of the data.
        ## TODO: Don't forget to pass the sampler object to the DataLoader.
        # Use multiple processes for loading data.
        num_workers=args.train_num_workers,
        # Use pinned memory on GPUs for faster device-copy.
        pin_memory=True,
        persistent_workers=args.train_num_workers > 0,
    )
    valid_dset = torch.utils.data.DataLoader(
        valid_dset,
        batch_size=args.batch_size,
        # Uses the distributed sampler to ensure each process gets a different subset of the data.
        ## TODO: Don't forget to pass the sampler object to the DataLoader.
        num_workers=args.valid_num_workers,
        # Use pinned memory on GPUs for faster device-copy.
        pin_memory=True,
        persistent_workers=args.valid_num_workers > 0,
    )
    test_dset = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.batch_size,
        # Uses the distributed sampler to ensure each process gets a different subset of the data.
        ## TODO: Don't forget to pass the sampler object to the DataLoader.
        # Use pinned memory on GPUs for faster device-copy.
        pin_memory=True,
    )
    return train_dset, valid_dset, test_dset


def train_batch(opt, model, loss_func, features, labels):
    """Train the model on a batch and return the global loss."""
    model.train()
    opt.zero_grad(set_to_none=True)

    preds = model(features)
    loss = loss_func(preds, labels)
    loss.backward()
    opt.step()
    # Obtain the global average loss.
    loss_avg = all_reduce_avg(loss)
    return loss_avg.item()


def test_model(model, loss_func, test_dset, device):
    """Evaluate the model on an evaluation set and return the global
    loss over the entire evaluation set.
    """
    model.eval()
    with torch.no_grad():
        loss = 0
        for (i, (features, labels)) in enumerate(test_dset):
            features = features.to(device)
            labels = labels.to(device)

            preds = model(features)
            loss += loss_func(preds, labels)
        loss /= len(test_dset)
    
    ## TODO:
    # Obtain the global average loss.

    return loss.item()


def main():
    args = parse_args()

    ## TODO: 
    # 1. Add code to initialize the distributed training environment.
    # 2. Get the identifier of each process within a node
    # 3. Get the global identifier of each process within the distributed system.
    # 4. Create a torch.device object that represents the GPU to be used by this process.
    # 5. Set the default CUDA device for the current process, ensuring all subsequent CUDA operations are performed on the specified GPU device.
    # 6. Set the random number generator initialization value for each process.

    train_dset, valid_dset, test_dset = prepare_datasets(args, device)

    model = torchvision.models.resnet50(weights=None)
    model = model.to(device)

    ## TODO: 
    # Wraps the model in a DistributedDataParallel (DDP) module or FullyShardedDataParallel (FSDP) module to parallelize the training across multiple GPUs.
   

    # model_checkpointing.load_model_checkpoint(model, 'model-final.pt')
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Maximum value of default dtype.
    min_valid_loss = torch.finfo(torch.get_default_dtype()).max
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        train_dset.sampler.set_epoch(epoch)

        for (i, (features, labels)) in enumerate(train_dset):
            features = features.to(device)
            labels = labels.to(device)
            
            opt.zero_grad(set_to_none=True)

            preds = model(features)
            loss = loss_func(preds, labels)
            loss.backward()
            opt.step()
            
            ## TODO:
            # Obtain the global average loss.
            
            if i % 10 == 0:
                ## TODO:
                # Replace print with the utility function to print messages only from rank 0.
                print(f'[{epoch}/{args.epochs}; {i}] loss: {loss:.5f}')

 
        valid_loss = test_model(model, loss_func, valid_dset, device)
        
        ## TODO:
        # Replace print with the utility function to print messages only from rank 0.
        print(f'[{epoch}/{args.epochs}; {i}] valid loss: {valid_loss:.5f}')
        
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            
            ## TODO:
            # Replace the following line with the utility function to save the model.
            torch.save(model, 'model-best.pt')

            
    end_time = time.perf_counter()
    
    ## TODO:
    # Replace print with the utility function to print messages only from rank 0.
    print('Finished training after', end_time - start_time, 'seconds.')
    
    test_loss = test_model(model, loss_func, test_dset, device)

    ## TODO:
    # Replace print with the utility function to print messages only from rank 0.
    print('Final test loss:', test_loss)
    
    ## TODO:
    # Replace the following line with the utility function to save the model.
    torch.save(args.save_model_opt, model, rank, save_optimizer=args.save_optimizer)


if __name__ == '__main__':
    main()
