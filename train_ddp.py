import argparse
import functools
import os
import time

import torch
from torch.distributed import checkpoint as dist_checkpoint
from torch.distributed import fsdp
import torchvision


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

    args = parser.parse_args()
    return args


functools.lru_cache(maxsize=None)
def is_root_process():
    """Return whether this process is the root process."""
    return torch.distributed.get_rank() == 0


def print0(*args, **kwargs):
    """Print something only on the root process."""
    if is_root_process():
        print(*args, **kwargs)


def save0(*args, **kwargs):
    """Pass the given arguments to `torch.save`, but only on the root
    process.
    """
    # We do *not* want to write to the same location with multiple
    # processes at the same time.
    if is_root_process():
        torch.save(*args, **kwargs)




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

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dset,
        shuffle=True,
        seed=args.seed,
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dset)

    train_dset = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        # Use multiple processes for loading data.
        num_workers=args.train_num_workers,
        # Use pinned memory on GPUs for faster device-copy.
        pin_memory=True,
        persistent_workers=args.train_num_workers > 0,
    )
    valid_dset = torch.utils.data.DataLoader(
        valid_dset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        num_workers=args.valid_num_workers,
        # Use pinned memory on GPUs for faster device-copy.
        pin_memory=True,
        persistent_workers=args.valid_num_workers > 0,
    )
    test_dset = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.batch_size,
        sampler=test_sampler,
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
    # Obtain the global average loss.
    torch.distributed.all_reduce(loss, torch.distributed.ReduceOp.AVG)
    return loss.item()


def main():
    args = parse_args()

    # Initializes a communication group using 'nccl' as the backend for GPU communication.
    torch.distributed.init_process_group(backend='nccl')

    # Get the identifier of each process within a node
    local_rank = int(os.getenv('LOCAL_RANK'))
    # Get the global identifier of each process within the distributed system
    rank = int(os.environ['RANK'])
    # Creates a torch.device object that represents the GPU to be used by this process.
    device = torch.device('cuda', local_rank)
    # Sets the default CUDA device for the current process, 
    # ensuring all subsequent CUDA operations are performed on the specified GPU device.
    torch.cuda.set_device(device)

    # Different random seed for each process.
    torch.random.manual_seed(args.seed + torch.distributed.get_rank())

    train_dset, valid_dset, test_dset = prepare_datasets(args, device)

    model = torchvision.models.resnet50(weights=None)
    model = model.to(device)

    # Wraps the model in a DistributedDataParallel (DDP) module to parallelize the training across multiple GPUs.
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
    )

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
            
            # Obtain the global average loss.
            torch.distributed.all_reduce(loss, torch.distributed.ReduceOp.AVG)

            if i % 10 == 0:
                print0(f'[{epoch}/{args.epochs}; {i}] loss: {loss:.5f}')


        valid_loss = test_model(model, loss_func, valid_dset, device)
        print0(f'[{epoch}/{args.epochs}; {i}] valid loss: {valid_loss:.5f}')

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            save0(model, 'model-best-ddp.pt')

            
    end_time = time.perf_counter()
    print0('Finished training after', end_time - start_time, 'seconds.')
    test_loss = test_model(model, loss_func, test_dset, device)

    print0('Final test loss:', test_loss)
    save0(model, 'model-final-ddp')


if __name__ == '__main__':
    main()
