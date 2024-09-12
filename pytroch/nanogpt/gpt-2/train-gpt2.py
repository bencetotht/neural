import os
from dataclasses import dataclass
import inspect
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # key, query, value for all heads, in one batch
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # output projection
        self.c_proj.NANOGPT_SCALE_INIT = 1 # for normalizing std
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # masking

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) # query, key, value
        # making number of heads (nh) into a batch dimension
        # nh = number of heads, hs = head size, C = number of channels (nh * hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention 
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) => (B, nh, T, hs)

        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # concat operation - re-assemble all head outputs

        y = self.c_proj(y) # output projection
        return y

class ManualGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi ) * (x + 0.044715 * torch.pow(x, 3.0))))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # tanh used in gpt-2
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # FFN - feed forward network

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50000 merges + 256 bytes + 1 end of text token
    n_layer: int = 6 # number of layers
    n_head: int = 6 # number of heads
    n_embd: int = 384 # dimension of transformer / embedding dimension

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict( # index into submodules using keys
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weights of token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # weights of positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden - modulelist = index using integers
            ln_f = nn.LayerNorm(config.n_embd), # layernorm - used in GPT-2 model
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # classifier - language model head

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # copying all weight data
        
        # init params
        self.apply(self._init_weights) # iterates through all of the sub-modules
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # 2x because two blocks
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): # token indicies
        B, T = idx.size() # idx shape: (B, T)
        assert T <= self.config.block_size, f"Cannot forward sequence length {T}, block size {B}"
        # forwarding token and pos embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # iterarting from 0 to T | (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # forward final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # getting all the params that require grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # creating optim groups, 2D parameters will be weight decayed
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters')
        print(f'num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters')
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # only in later versions of pytorch
        use_fused = fused_available and device == 'cuda'
        print(f'using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes): # batch size, time
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B*T)} batches')

        self.current_position = self.B * self.T * self.process_rank # setting state
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T +1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B*T * self.num_processes # advance position
        # if next batch would be out of bounds
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


# setting up Distributed Data Parallel
ddp = int(os.environ.get('RANK', -1)) != -1 # checking if ddp run
if ddp:
    assert torch.cuda.is_available(), 'DDP wont work without CUDA'
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # rank of gpu on a single node
    ddp_world_size = int(os.environ['WORLD_SIZE']) # number of processes
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # process of logging, checkpointing
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 'cpu'    

print(f'Torch running on {device} with version: {torch.__version__}')

# eval
torch.manual_seed(1337)
if device == 'cuda': 
    torch.cuda.manual_seed(1337)

# gradient accumulation
total_batch_size = 524288
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "total batch size is not divisible by B * T * dpp_world_size"
grad_accum_steps = total_batch_size // (B*T * ddp_world_size)
if master_process:
    print(f'total desired batch size: {total_batch_size}')
    print(f'=> calculated gradient accumulation steps: {grad_accum_steps}')

# get data batch
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size) # batch size, time

torch.set_float32_matmul_precision('high') # setting dtype to tensorflow32

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # getting raw model

# learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it): # iteration
    # 1. linear warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 3. it > max steps, minimum lr
    if it > max_steps:
        return min_lr
    # 2. between: cosine decay down to min lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # start at 0 goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=[0.9, 0.95], eps=1e-8) # 3e-4 commonly used for debugging | betas & eps used in gpt-3
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device=device)

for i in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()

    loss_acum = 0.0
    for microstep in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # setting dtype of logits to bfloat16
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_acum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (microstep == grad_accum_steps - 1) # only sync backward on the last microstep
        loss.backward()
    if ddp:
        dist.all_reduce(loss_acum, op=dist.ReduceOp.AVG) # making it identical
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clipping global norm of the gradient at 1.0
    
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0) * 1000
    tokens_per_second = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1-t0)
    if master_process:
        print(f'step {i}, loss: {loss_acum.item():.4f}, learning rate: {lr:.4e}, norm: {norm:.4f}, time: {dt:.2f}ms, tokens: {tokens_per_second:.2f}')

if ddp:
    destroy_process_group()

import sys; sys.exit(0)

num_return_sequences = 5
max_length = 30

# prefix tokens
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# sampling
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # getting logits from model | (B,T,vocab_size)
        logits = logits[:, -1, :] # taking logits at the last position | (B, vocab_size)
        probs = F.softmax(logits)
        # top-k sampling
        topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1) # selecting token from topk_probs | (B,1)
        xcol = torch.gather(topk_indicies, -1, ix) # gather indicies | (B,1)
        # appending to sequence
        x = torch.cat((x, xcol), dim=1)

# print generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)