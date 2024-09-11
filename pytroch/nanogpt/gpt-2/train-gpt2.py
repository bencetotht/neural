from dataclasses import dataclass
import math
from matplotlib.pylab import logistic
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 'cpu'
print(f'Torch running on {device} with version: {torch.__version__}')

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
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) => (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # concat operation - re-assemble all head outputs

        y = self.c_proj(y) # output projection
        return y

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

class DataLoaderLite:
    def __init__(self, B, T): # batch size, time
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B*T)} batches')

        self.current_position = 0 # setting state
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+ B*T +1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B*T # advance position
        # if next batch would be out of bounds
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# eval
num_return_sequences = 5
max_length = 30

model = GPT(GPTConfig())
model.eval()
model.to(device)

torch.manual_seed(1337)
if device == 'cuda': 
    torch.cuda.manual_seed(1337)
# get data batch
train_loader = DataLoaderLite(4, 32) # batch size, time

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # 3e-4 commonly used for debugging
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f'step {i}, loss: {loss.item()}')

import sys; sys.exit(0)

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