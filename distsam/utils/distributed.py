import os
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = int(getattr(args, "local_rank", 0))

    args.distributed = args.world_size > 1
    if not args.distributed:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training with NCCL requires CUDA.")
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend=getattr(args, "dist_backend", "nccl"),
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
    )
    try:
        if torch.cuda.is_available():
            dist.barrier(device_ids=[args.local_rank])
        else:
            dist.barrier()
    except TypeError:
        dist.barrier()


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_main(state, path):
    if is_main_process():
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)


def reduce_dict(input_dict: Dict[str, torch.Tensor], average: bool = True):
    if not is_dist_avail_and_initialized():
        return input_dict
    names = []
    values = []
    for key in sorted(input_dict.keys()):
        names.append(key)
        value = input_dict[key]
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=torch.float32, device="cuda")
        values.append(value.detach())
    values = torch.stack(values, dim=0)
    dist.all_reduce(values)
    if average:
        values /= get_world_size()
    return {key: value for key, value in zip(names, values)}


def all_reduce_scalar(value, average: bool = True, device=None):
    if torch.is_tensor(value):
        tensor = value.detach().clone().float()
    else:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
        tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor)
        if average:
            tensor /= get_world_size()
    return tensor.item()


def reduce_metric_sums(metric_dict: Dict[str, torch.Tensor]):
    if not is_dist_avail_and_initialized():
        return metric_dict
    reduced = {}
    for key, value in metric_dict.items():
        tensor = value.detach().clone()
        dist.all_reduce(tensor)
        reduced[key] = tensor
    return reduced


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()
