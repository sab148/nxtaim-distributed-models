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

    train_dset = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size,
        # Use multiple processes for loading data.
        num_workers=args.train_num_workers,
        # Use pinned memory on GPUs for faster device-copy.
        pin_memory=True
    )
    valid_dset = torch.utils.data.DataLoader(
        valid_dset,
        batch_size=args.batch_size,
        num_workers=args.valid_num_workers,
        # Use pinned memory on GPUs for faster device-copy.
        pin_memory=True
    )
    test_dset = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.batch_size,
        # Use pinned memory on GPUs for faster device-copy.
        pin_memory=True
    )
    return train_dset, valid_dset, test_dset




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

    return loss.item()

def main():
    args = parse_args()

    torch.random.manual_seed(args.seed)

    device = torch.device('cuda')
    train_dset, valid_dset, test_dset = prepare_datasets(args, device)

    model = torchvision.models.resnet50(weights=None)
    model = model.to(device)
    
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    min_valid_loss = torch.finfo(torch.get_default_dtype()).max
    
    print('Starting training...')  
    start_time = time.perf_counter()
    
    
    for epoch in range(args.epochs):
        model.train()

        for (i, (features, labels)) in enumerate(train_dset):
            features = features.to(device)
            labels = labels.to(device)

            opt.zero_grad(set_to_none=True)
            
            preds = model(features)

            loss = loss_func(preds, labels)
            loss.backward()
            
            opt.step()

            if i % 10 == 0:
                print(f'[{epoch}/{args.epochs}; {i}] loss: {loss:.5f}')

        valid_loss = test_model(model, loss_func, valid_dset, device)
        print(f'[{epoch}/{args.epochs}; {i}] valid loss: {valid_loss:.5f}')

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model, 'model-best')

    end_time = time.perf_counter()
    
    print('Finished training after', end_time - start_time, 'seconds.')
    test_loss = test_model(model, loss_func, test_dset, device)

    print('Final test loss:', test_loss)
    torch.save(model, 'model-final')


if __name__ == '__main__':
    main()
