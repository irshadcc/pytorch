import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.distributed_c10d import _get_default_group
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


# CHECKPOINT_DIR = f"~/{os.environ['LOGNAME']}/checkpoint"


# class Model(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         torch.manual_seed(0)
#         self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
#         self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
#         self.net3 = nn.Linear(32, 64)
#         self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

#     def forward(self, x):
#         return self.net4(self.net3(self.net2(self.net1(x))))

#     def get_input(self):
#         return torch.rand(8, 8, device="cuda")


# def _make_stateful(model, optim):
#     _patch_model_state_dict(model)
#     _patch_optimizer_state_dict(model, optimizers=optim)


# def _train(model, optim, train_steps=1):
#     torch.manual_seed(0)
#     loss = None
#     for _ in range(train_steps):
#         loss = model(model.get_input()).sum()
#         loss.backward()
#         optim.step()
#         optim.zero_grad()

#     return loss


# def _init_model(device, world_size):
#     device_mesh = init_device_mesh(device, (world_size,))
#     model = Model().cuda()
#     model = FSDP(
#         model,
#         device_mesh=device_mesh,
#         use_orig_params=True,
#     )
#     optim = torch.optim.Adam(model.parameters(), lr=0.1)
#     _make_stateful(model, optim)

#     return model, optim


def run(rank, world_size, device="cuda"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["PYTORCH_API_USAGE_STDERR"] = "1"
    dist.init_process_group(rank=rank, world_size=world_size)
    tensor = torch.tensor([1.0, 2.0, 3.0])
    # Perform all-reduce sum
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)



    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 1
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )


world_size=1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["PYTORCH_API_USAGE_STDERR"] = "1"
dist.init_process_group(rank=0, world_size=world_size)
parameter = nn.Parameter()
tensor = torch.tensor([1.0, 2.0, 3.0])
# Perform all-reduce sum
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
dist.destroy_process_group()
