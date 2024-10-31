import io
import os
import math
import time
import json
import copy
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import subprocess
import torch
import torch.distributed as dist
# from torch._six import inf
from torch import inf
import random
import torch.nn.functional as F

from tensorboardX import SummaryWriter


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)

        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # assert torch.distributed.is_initialized()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))

def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data

def load_box_encoder(args):
    model = torch.load(args.box_encoder_init_ckpt, map_location='cpu')
    model = torch.load(args.box_encoder_init_ckpt, map_location='cpu')
    for param in model.parameters():
        param.requires_grad = False

    return model

class BoxPretrainingModel(torch.nn.Module):
    def __init__(self, boxEncoder, videoEncoder):
        super(BoxPretrainingModel, self).__init__()
        self.boxEncoder = boxEncoder
        self.videoEncoder = videoEncoder
        
    def forward(self, box_src, box_tgt, box_src_padding_mask, frames):
        box_outputs = self.boxEncoder(box_src, box_tgt, box_src_padding_mask)
        video_outputs = self.videoEncoder(frames)
        return box_outputs, video_outputs

'''
Following code from https://github.com/FingerRec/OA-Transformer/tree/main
official code release for OA-Transformer,
Object-aware Video-language Pre-training for Retrieval (CVPR 2022)
'''

# From OATrans/model/model.py
# To compute similarity between visual and box features
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

# From OATrans/model/loss.py
class NormSoftmaxLoss(torch.nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        """Assumes input x is similarity matrix of N x M \in [-1, 1], 
            computed using the cosine similarity between normalised vectors"""
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j


# From OATrans/trainer/trainer_dist.py
# 
class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(n_gpu)]
        dist.all_gather(output, tensor)
        ctx.local_rank = args.local_rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.local_rank : ctx.batch_size * (ctx.local_rank + 1)],
            None, None,
        )

class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )


def set_requires_grad_for(model, layer_list):
    for name, param in model.named_parameters():
        if any(x in name for x in layer_list):
            param.requires_grad = True

# Function to set requires_grad for varying number of layers
def set_param_requires_grad(model, folder_name):
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    if "sact_uniSpl_FTlayers_H-FCN-1B" in folder_name:
        set_requires_grad_for(model, ["head", "fc_norm", "blocks.11"])
    elif "sact_uniSpl_FTlayers_H-FCN-2B" in folder_name:
        set_requires_grad_for(model, ["head", "fc_norm", "blocks.11", "blocks.10"])
    elif "sact_uniSpl_FTlayers_H-FCN-3B" in folder_name:
        set_requires_grad_for(model, ["head", "fc_norm", "blocks.11", "blocks.10", "blocks.9"])
    elif "sact_uniSpl_FTlayers_H-FCN-4B" in folder_name:
        set_requires_grad_for(model, ["head", "fc_norm", "blocks.11", "blocks.10", "blocks.9", "blocks.8"])
    elif "sact_uniSpl_FTlayers_H-FCN-5B" in folder_name:
        set_requires_grad_for(model, ["head", "fc_norm", "blocks.11", "blocks.10", "blocks.9", "blocks.8", "blocks.7"])
    elif "sact_uniSpl_FTlayers_H-FCN" in folder_name:
        set_requires_grad_for(model, ["head", "fc_norm"])
    elif "sact_uniSpl_FTlayers_H" in folder_name:
        set_requires_grad_for(model, ["head"])
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True


# Copy block and set linears to zero - Block Expansion
def expand_blocks(model, block_ids):
    for id in block_ids:
        model.blocks[id] = torch.nn.Sequential(model.blocks[id], copy.deepcopy(model.blocks[id]))

        # Set parameters of Linears before identity addition to zero
        with torch.no_grad():
            model.blocks[id][1].attn.proj.weight = torch.nn.Parameter(
                torch.zeros_like(model.blocks[id][1].attn.proj.weight))
            model.blocks[id][1].attn.proj.bias = torch.nn.Parameter(
                torch.zeros_like(model.blocks[id][1].attn.proj.bias))
            model.blocks[id][1].mlp.fc2.weight = torch.nn.Parameter(
                torch.zeros_like(model.blocks[id][1].mlp.fc2.weight))
            model.blocks[id][1].mlp.fc2.bias = torch.nn.Parameter(
                torch.zeros_like(model.blocks[id][1].mlp.fc2.bias))

# Function for block expansion, based on output directory name
def block_expansion(model, folder_name):
    if "BExp" in folder_name:
        for name, param in model.named_parameters():
            param.requires_grad = False

        block_ids = [int(x) for x in folder_name.split("-")[1:]]
        expand_blocks(model, block_ids)

        block_names = [f"blocks.{x}.1" for x in folder_name.split("-")[1:]]
        set_requires_grad_for(model, block_names)


# Define adapter layer 
class AdapterLayer(torch.nn.Module):
    def __init__(self, dim, reduced_dim, scaling):
        super(AdapterLayer, self).__init__()
        self.adapter_down = torch.nn.Linear(dim, reduced_dim)
        self.adapter_up = torch.nn.Linear(reduced_dim, dim)
        
        # Zeroing out adapter_up weights so that initial output remains the same
        self.adapter_up.weight = torch.nn.Parameter(torch.zeros_like(self.adapter_up.weight))
        self.adapter_up.bias = torch.nn.Parameter(torch.zeros_like(self.adapter_up.bias))

        # Set a scaling factor based on reduced_dim
        self.scaling = scaling / reduced_dim

    def forward(self, x):
        x = self.adapter_down(x)
        x = torch.nn.functional.relu(x)
        x = self.scaling * self.adapter_up(x)
        return x

class BlockWithAdapter(torch.nn.Module):
    def __init__(self, original_block, reduced_dim, scaling):
        super(BlockWithAdapter, self).__init__()
        self.drop_path = original_block.drop_path
        self.attn =  original_block.attn
        self.norm1 =  original_block.norm1
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        self.gamma_1 = original_block.gamma_1
        self.gamma_2 = original_block.gamma_2
        
        # Add the adapter in parallel
        self.adapter = AdapterLayer(original_block.mlp.fc1.in_features, reduced_dim, scaling)    

    def forward(self, x):
        # Run the adapter in parallel
        adapter_output = self.adapter(x)

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)) + adapter_output)
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)) + adapter_output)    
            
        return x


# Function for adding adapters, following AdaptFormer
def apply_adapters(model, folder_name):
    if "ADAP" in folder_name:
        reduced_dim = int(folder_name.split("-")[-1])
        scaling = float(folder_name.split("-")[-2])
        for name, param in model.named_parameters():
            param.requires_grad = False

        layer_list = []
        for block_id in range(len(model.blocks)):
            model.blocks[block_id] = BlockWithAdapter(model.blocks[block_id], reduced_dim, scaling)
            layer_list.append(f"model.blocks[{block_id}].adapter")

        set_requires_grad_for(model, layer_list)


class AttnWithLoRA(torch.nn.Module):
    def __init__(self, original_attn, reduced_dim, scaling):
        super(AttnWithLoRA, self).__init__()
        self.num_heads = original_attn.num_heads
        self.scale =  original_attn.scale
        self.qkv =  original_attn.qkv
        self.q_bias =  original_attn.q_bias
        self.v_bias =  original_attn.v_bias
        self.attn_drop =  original_attn.attn_drop
        self.proj =  original_attn.proj
        self.proj_drop =  original_attn.proj_drop
        
        # Add lora layers for qkv and proj
        self.qkv_lora_down = torch.nn.Linear(original_attn.qkv.in_features, reduced_dim*3, bias=False)
        self.qkv_lora_up = torch.nn.Linear(reduced_dim*3, original_attn.qkv.out_features, bias=False)    
        self.proj_lora_down = torch.nn.Linear(original_attn.proj.in_features, reduced_dim, bias=False)
        self.proj_lora_up = torch.nn.Linear(reduced_dim, original_attn.proj.out_features, bias=False)

        # # Zeroing out up weights so that initial output remains the same
        self.qkv_lora_up.weight = torch.nn.Parameter(torch.zeros_like(self.qkv_lora_up.weight))
        self.proj_lora_up.weight = torch.nn.Parameter(torch.zeros_like(self.proj_lora_up.weight))

        # Set a scaling factor based on reduced_dim
        self.lora_scaling = scaling / reduced_dim

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Add qkv_lora with scaling
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias) + \
                self.lora_scaling * self.qkv_lora_up(self.qkv_lora_down(x))
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # Add proj_lora with scaling
        x = self.proj(x) + self.lora_scaling * self.proj_lora_up(self.proj_lora_down(x))
        x = self.proj_drop(x)

        return x
            
# Function for adding LoRA weight matrices to 
# qkv and proj layers in attn in each block
def apply_lora(model, folder_name):
    if "LoRA" in folder_name:
        reduced_dim = int(folder_name.split("-")[-1])
        scaling = float(folder_name.split("-")[-2])
        for name, param in model.named_parameters():
            param.requires_grad = False

        layer_list = []
        for block_id in range(len(model.blocks)):
            model.blocks[block_id].attn = AttnWithLoRA(model.blocks[block_id].attn, reduced_dim, scaling)
            layer_list.append(f"model.blocks[{block_id}].attn.qkv_lora_down")
            layer_list.append(f"model.blocks[{block_id}].attn.qkv_lora_up")
            layer_list.append(f"model.blocks[{block_id}].attn.proj_lora_down")
            layer_list.append(f"model.blocks[{block_id}].attn.proj_lora_up")

        set_requires_grad_for(model, layer_list)