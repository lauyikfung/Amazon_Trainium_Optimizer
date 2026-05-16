import os
import time
import math
import pickle
from contextlib import nullcontext
from collections import deque

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

try:
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend
except ImportError:
    xr = None
    xm = None

from model import GPTConfig, GPT
import sys
from ast import literal_eval
# code: single core: python3 train_mars_m.py     --device=xla     --compile=False     --optimizer_name=mars-m     --batch_size=4   \
# --gradient_accumulation_steps=1     --block_size=512     --n_embd=384     --n_layer=3     --n_head=6     --max_iters=200     \
# --eval_interval=100000     --eval_iters=1     --xla_save_checkpoints=False --learning_rate=1e-2 --warmup_iters=20 --xla_optimizer_cpu_step=True
# multi-core: torchrun --nproc_per_node=2 train_mars_m.py ...

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
data_path = "./data"
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
initial_steps = 100
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# optimizer
optimizer_name = 'mars-m' 
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.95
beta2 = 0.99
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
ns_steps = 5
interval = 10
variant = 4 
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'xla' # examples: 'xla' for Trainium/Neuron, 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
xla_skip_initial_eval = True # avoid compiling eval and training graphs back-to-back on Trainium startup
xla_save_checkpoints = False # XLA checkpoint materialization can stall low-memory Trainium runs
xla_optimizer_cpu_step = True # offload only MARS-M matrix updates; AdamW backup stays on XLA
scale_attn_by_inverse_layer_idx = True
# learning rate schedule
schedule='cosine'
scheme='exact'
gamma=0.025
clip_c=False
lr_1d=3e-3
is_approx=True
betas_1d=(0.9, 0.95)
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

def want_xla_device(device_config):
    return str(device_config).lower() in ('xla', 'neuron', 'trainium')

use_xla = want_xla_device(device)
requested_compile = compile
if use_xla:
    if xm is None:
        raise ImportError("device='xla' requires torch_xla/torch-neuronx. Install the AWS Neuron PyTorch stack.")
    if '--optlevel' not in os.environ.get('NEURON_CC_FLAGS', ''):
        os.environ['NEURON_CC_FLAGS'] = (os.environ.get('NEURON_CC_FLAGS', '') + ' --optlevel=1').strip()
    device = xm.xla_device()
    backend = 'xla'
    compile = False
    if xla_skip_initial_eval:
        print("Trainium/XLA: skipping the initial step-0 eval by default to avoid startup compile stalls.")
    if dtype == 'float16':
        print("float16 is not supported for this Trainium path; using bfloat16 instead.")
        dtype = 'bfloat16'
elif str(device).startswith('cuda') and not torch.cuda.is_available():
    if xm is not None:
        print("CUDA is not available; falling back to XLA/Neuron device.")
        use_xla = True
        if '--optlevel' not in os.environ.get('NEURON_CC_FLAGS', ''):
            os.environ['NEURON_CC_FLAGS'] = (os.environ.get('NEURON_CC_FLAGS', '') + ' --optlevel=1').strip()
        device = xm.xla_device()
        backend = 'xla'
        compile = False
        if xla_skip_initial_eval:
            print("Trainium/XLA: skipping the initial step-0 eval by default to avoid startup compile stalls.")
        if dtype == 'float16':
            print("float16 is not supported for this Trainium path; using bfloat16 instead.")
            dtype = 'bfloat16'
    else:
        raise RuntimeError("CUDA is not available and torch_xla is not installed. Set --device=cpu or install torch-neuronx.")

def xla_mark_step():
    if use_xla:
        xm.mark_step()

def xla_global_ordinal():
    if xr is not None and hasattr(xr, 'global_ordinal'):
        return xr.global_ordinal()
    if hasattr(xm, 'get_ordinal'):
        return xm.get_ordinal()
    return int(os.environ.get('RANK', 0))

def xla_local_ordinal():
    if xr is not None and hasattr(xr, 'local_ordinal'):
        return xr.local_ordinal()
    if hasattr(xm, 'get_local_ordinal'):
        return xm.get_local_ordinal()
    return int(os.environ.get('LOCAL_RANK', 0))

def xla_world_size():
    if xr is not None and hasattr(xr, 'world_size'):
        return xr.world_size()
    if hasattr(xm, 'xrt_world_size'):
        return xm.xrt_world_size()
    return int(os.environ.get('WORLD_SIZE', 1))

def save_checkpoint(checkpoint, path):
    if use_xla:
        xm.save(checkpoint, path)
    else:
        torch.save(checkpoint, path)

def should_save_checkpoint():
    return (not use_xla) or xla_save_checkpoints

def optimizer_step(optimizer):
    if use_xla:
        if getattr(optimizer, 'step_callback', None) is not None:
            grads = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            world_size = xla_world_size()
            if grads and world_size > 1:
                xm.all_reduce(xm.REDUCE_SUM, grads, scale=1.0 / world_size)
            xla_mark_step()
            optimizer.step()
        else:
            xm.optimizer_step(optimizer)
    else:
        optimizer.step()

def cpu_optimizer_step(xla_model, cpu_model, cpu_optimizer):
    return cpu_optimizer_step_for_pairs(
        list(zip(xla_model.parameters(), cpu_model.parameters())),
        cpu_optimizer,
    )

def cpu_optimizer_step_for_pairs(param_pairs, cpu_optimizer):
    grads = [xla_param.grad for xla_param, _ in param_pairs if xla_param.grad is not None]
    world_size = xla_world_size()
    if grads and world_size > 1:
        xm.all_reduce(xm.REDUCE_SUM, grads, scale=1.0 / world_size)
    xla_mark_step()
    with torch.no_grad():
        for xla_param, cpu_param in param_pairs:
            if xla_param.grad is None:
                cpu_param.grad = None
            else:
                cpu_param.grad = xla_param.grad.detach().cpu()
    cpu_optimizer.step()
    cpu_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for xla_param, cpu_param in param_pairs:
            xla_param.copy_(cpu_param.to(device=xla_param.device, dtype=xla_param.dtype))
    xla_mark_step()

class ExactAdamWBackup(torch.optim.Optimizer):
    def __init__(self, params, lr, wd, betas=(0.9, 0.95), eps=1e-8):
        super().__init__(params, dict(lr=lr, wd=wd, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            wd = group['wd']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                update = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-lr / scale)
        return loss

class NoOpGradScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer_step(optimizer)

    def update(self):
        pass

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    if use_xla:
        init_process_group('xla')
        ddp_rank = xla_global_ordinal()
        ddp_local_rank = int(os.environ.get('LOCAL_RANK', xla_local_ordinal()))
        device = xm.xla_device()
    else:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    if not use_xla:
        gradient_accumulation_steps *= 8 # simulate 8 gpus

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(5000 + seed_offset)
if str(device).startswith('cuda'):
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'xla' if use_xla else ('cuda' if 'cuda' in str(device) else 'cpu') # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join(data_path, dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu' if use_xla else device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(device_type == 'cuda' and dtype == 'float16')) if device_type == 'cuda' else NoOpGradScaler()
cpu_optimizer_model = None
xla_adamw_optimizer = None
cpu_muon_param_pairs = None
optimizer_model = model
if use_xla and optimizer_name == 'mars-m' and xla_optimizer_cpu_step:
    print("Trainium/XLA: running MARS-M matrix updates on CPU; AdamW backup stays on XLA.")
    gptconf = GPTConfig(**model_args)
    cpu_optimizer_model = GPT(gptconf)
    cpu_optimizer_model.load_state_dict(model.state_dict())
    cpu_optimizer_model.to('cpu')
    optimizer_model = cpu_optimizer_model
xla_muon_params = [
    p for name, p in model.named_parameters()
    if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name and "wpe" not in name and "wte" not in name
]
xla_adamw_params = [
    p for name, p in model.named_parameters()
    if p.ndim < 2 or "embed_tokens" in name or "lm_head" in name or "wpe" in name or "wte" in name
]
muon_params = [
    p for name, p in optimizer_model.named_parameters()
    if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name and "wpe" not in name and "wte" not in name
]
adamw_params = [] if cpu_optimizer_model is not None else [
    p for name, p in optimizer_model.named_parameters()
    if p.ndim < 2 or "embed_tokens" in name or "lm_head" in name or "wpe" in name or "wte" in name
]
if cpu_optimizer_model is not None:
    cpu_muon_param_pairs = list(zip(xla_muon_params, muon_params))
    xla_adamw_optimizer = ExactAdamWBackup(
        xla_adamw_params,
        lr=learning_rate,
        wd=weight_decay,
        betas=betas_1d,
        eps=1e-8,
    )
from optimizers.mars_m import MARS_M
if optimizer_name == 'mars-m':
    optimizer = MARS_M(lr=learning_rate, wd=weight_decay, muon_params=muon_params, adamw_params=adamw_params,
                       is_approx=is_approx, gamma=gamma, clip_c=clip_c, momentum=beta1,
                       ns_steps=ns_steps, adamw_betas=betas_1d,
                       step_callback=xla_mark_step if use_xla and not xla_optimizer_cpu_step else None)
elif optimizer_name == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                  betas=(beta1, beta2), foreach=False)
else:
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    del state_dict
    del checkpoint
# compile the model
if requested_compile and use_xla:
    print("torch.compile is disabled on Trainium/XLA; using torch_xla lazy compilation instead.")
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    print('DDP_used')
    if not use_xla:
        model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            xla_mark_step()
            if not torch.isfinite(loss).item():
                print(
                    f"non-finite {split} loss at eval batch {k}: "
                    f"loss={loss.item()}, "
                    f"logits finite={torch.isfinite(logits).all().item()}, "
                    f"logits min={logits.min().item()}, logits max={logits.max().item()}"
                )
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, schedule='cosine'):
    #ing rate schedule {schedule}")
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if schedule=='wsd':
        if it < 0.8 * max_iters:
            return learning_rate
        else:
            return learning_rate * (max_iters - it) / (max_iters * 0.2)
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    if schedule=='cosine':
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    elif schedule=='exp':
        coeff = np.power(0.9, 100 * decay_ratio)
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
#X, Y = get_batch('train') # fetch the very first batch
Xs=deque([])
Ys=deque([])
for micro_step in range(gradient_accumulation_steps):
    X, Y = get_batch('train')
    Xs.append(X)
    Ys.append(Y)
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if (ddp and not use_xla) else model # unwrap DDP container if needed
running_mfu = -1.0
clip_time = 0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num, schedule=schedule) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if xla_adamw_optimizer is not None:
        for param_group in xla_adamw_optimizer.param_groups:
            param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    do_eval = iter_num % eval_interval == 0 and master_process
    if use_xla and xla_skip_initial_eval and iter_num == 0 and not eval_only:
        do_eval = False
    if do_eval:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }, step=iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                if should_save_checkpoint():
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    save_checkpoint(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                else:
                    print("Trainium/XLA: checkpoint save skipped; set --xla_save_checkpoints=True to enable.")
        if iter_num > 0 and iter_num % (eval_interval * 5) == 0:
            if should_save_checkpoint():
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                save_checkpoint(checkpoint, os.path.join(out_dir, 'ckpt_'+str(iter_num)+'.pt'))
            else:
                print("Trainium/XLA: checkpoint save skipped; set --xla_save_checkpoints=True to enable.")
    if iter_num == 0 and eval_only:
        xla_mark_step()
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    minibatch_size = gradient_accumulation_steps
    X_cur = []
    Y_cur = []
    ## Update datasets
    for micro_step in range(minibatch_size):
        X_cur.append(Xs.popleft())
        Y_cur.append(Ys.popleft())
        X, Y = get_batch('train')
        Xs.append(X)
        Ys.append(Y)
    ## Calculate previous gradient with future batch data first, this information should be used at the next iteration.
    if scheme == 'exact' and not is_approx:
        ### Calculate the gradient again using the new batch
        for micro_step in range(gradient_accumulation_steps):
                if ddp and not use_xla:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                with ctx:
                    X = Xs[micro_step]
                    Y = Ys[micro_step]
                    logits, loss = model(X, Y)
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
                xla_mark_step()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if total_norm.item() > grad_clip:
                clip_time += 1  
        elif (grad_clip == 0.0) and (optimizer.gamma == 0.0):
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if total_norm.item() > 1.0:
                clip_time += 1
        ### Update the previous grad of the next iteration
        if hasattr(optimizer, 'update_previous_grad'):
            optimizer.update_previous_grad()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

    ## Calculate the gradient of the current batch
    for micro_step in range(minibatch_size):
        if ddp and not use_xla:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == minibatch_size - 1)
        with ctx:
            X = X_cur[micro_step]
            Y = Y_cur[micro_step]
            logits, loss = model(X, Y)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        xla_mark_step()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if total_norm.item() > grad_clip:
            clip_time += 1
    ### First update the current value of gradient
    #optimizer.update_current_grad() 
    # step the optimizer and scaler if training in fp16
    if cpu_optimizer_model is not None:
        cpu_optimizer_step_for_pairs(cpu_muon_param_pairs, optimizer)
        optimizer_step(xla_adamw_optimizer)
    else:
        scaler.step(optimizer)
        scaler.update()
    ### TODO: Clean the grad
    optimizer.zero_grad(set_to_none=True)
    if xla_adamw_optimizer is not None:
        xla_adamw_optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    if hasattr(optimizer, 'update_last_grad'):
        optimizer.update_last_grad()
    xla_mark_step()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        params = []
        for (name, p) in model.named_parameters():
            params.append(p)
        total_param_norm = 0
        for p in params:
            param_norm = p.data.norm(2)
            total_param_norm += param_norm.item() ** 2
        total_param_norm = total_param_norm ** 0.5
        momentum_norm = 0
        momentum_norm_sq = 0
        momentum_div = 0
        LL = len(optimizer.state_dict()['state'])
        for jj in range(LL):
            if 'exp_avg' in optimizer.state_dict()['state'][jj]:
                momentum_norm += (optimizer.state_dict()['state'][jj]['exp_avg'].detach().norm(2)) ** 2
            if 'momentum_buffer' in optimizer.state_dict()['state'][jj]:
                momentum_norm += (optimizer.state_dict()['state'][jj]['momentum_buffer'].detach().norm(2)) ** 2
            if 'moment2' in optimizer.state_dict()['state'][jj]:
                momentum_norm_sq += (optimizer.state_dict()['state'][jj]['moment2'].detach().norm(2)) ** 2
        momentum_norm = torch.sqrt(momentum_norm).item()
        try:
            momentum_norm_sq = torch.sqrt(momentum_norm_sq).item()
        except:
            momentum_norm_sq = 0
        momentum_div = momentum_norm/(np.sqrt(momentum_norm_sq)+1e-8)
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "param_norm": total_param_norm,
                "momentum_norm" : momentum_norm,
                "momentum_norm_sq": momentum_norm_sq,
                "momentum_div": momentum_div,
                "train/clip_rate": clip_time / (iter_num + 1)
            }, step=iter_num)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
