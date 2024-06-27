from pathlib import Path
from datetime import datetime
import torch
import time

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)


from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist


def save_model_checkpoint(model, save_dir, rank):
    """saving model via rank0 cpu streaming and full_state_dict"""

    fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    # saving with rank0 cpu
    with FSDP.state_dict_type(
        model, 
        StateDictType.FULL_STATE_DICT, 
        fullstate_save_policy
    ):
        cpu_state = model.state_dict()

    if rank == 0:    
        torch.save(cpu_state, save_dir)


def save_optimizer_checkpoint(model, optimizer, save_dir, rank):
    """save optimizer state via full state dict"""
    # pull all sharded optimizer states to rank0 cpu...

    optim_state = FSDP.full_optim_state_dict(model, optimizer)
    torch.save(optim_state, save_dir)

def save_distributed_model_checkpoint(model, save_dir, rank):

    localstate_save_policy = LocalStateDictConfig(offload_to_cpu=False)

    with FSDP.state_dict_type(
        model,
        StateDictType.LOCAL_STATE_DICT,
        localstate_save_policy
    ):
        state_dict = model.state_dict()

    # write out distributed checkpoint
    save_state_dict(
        state_dict, 
        FileSystemWriter(save_dir)
    )



def save_model_and_optimizer_sharded(model, save_dir, rank, optim=None):
    """Save model and optimizer via sharded_state_dict to save_dir"""

    shardedstate_save_policy = ShardedStateDictConfig(offload_to_cpu=False)

    with FSDP.state_dict_type(
        model, 
        StateDictType.SHARDED_STATE_DICT,
        shardedstate_save_policy
    ):

        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(save_dir),
            planner=DefaultSavePlanner(), 
        )

        
def load_model_checkpoint(model, load_dir):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    model_checkpoint = torch.load(load_dir)
    
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)


def load_optimizer_checkpoint(model, load_dir):
    """load an fdsp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """

    if rank == 0:
        full_osd = torch.load(load_dir)

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)


def load_distributed_model_checkpoint(model, load_dir):

    localstate_load_policy = LocalStateDictConfig(offload_to_cpu=False)

    with FSDP.state_dict_type(
        model,
        StateDictType.LOCAL_STATE_DICT,
        localstate_load_policy
    ):
        state_dict = model.state_dict()
        load_state_dict(state_dict, FileSystemReader(load_dir))
        model.load_state_dict(state_dict)


def load_model_sharded(model, load_dir):

    shardedstate_load_policy = ShardedStateDictConfig(offload_to_cpu=False)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, shardedstate_load_policy):
        checkpoint = model.state_dict()
      
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=FileSystemReader(load_dir),
        )

        model.load_state_dict(checkpoint)




