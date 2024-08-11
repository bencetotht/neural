from turtle import forward
from numpy import block
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 64 # number of inputs processed in parallel
block_size = 256 # maximum context length
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f'length of dataset: {len(text)}')

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f'{vocab_size=}')

stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encode into integers
decode = lambda d: ''.join([itos[c] for c in d]) # decode integers
data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)


def get_batch(split):
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # loading into device
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval() # setting model into evalution phase
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = m(X,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train() # setting back to train phase
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.droput = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape # batch, time, channels
        # key = what you have, query = what youre looking for
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)

        # transposing last 2 dimensions - (B, T, 16) @ (B, 16, T) -> (B, T, T)
        weights = q @ k.transpose(-2, -1) * C ** -0.5
        weights = weights.masked_fill(self.tril[:T][:T] == 0, float('-inf')) # replcaing zeros to -inf, nodes cant talk with eachother
        weights = F.softmax(weights, dim=1) # avg of past and current token
        weights = self.droput(weights)
        v = self.value(x)
        out = weights @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h[x] for h in self.heads], dim=1) # concat on chanel dimension
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential([
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout) # masking out some part of the neurons, creating subnets
        ])

    def forward(self, x):
        return self.net(x)

# transformer block
class Block(nn.Module): 
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa_heads = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # layer normalization, same as batch but on different dimension
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # so vectors know their position in space
        # self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8 dimensional self-attention
        # self.ffw = FeedForward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=4) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C) tensor (batch, time, channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        # x = self.sa_heads(x) # (B,T,C) - applying self-attention head
        # x = self.ffw(x) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
        # converting into 2d
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # negative log likelihood, (B,C,T)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx(B,T): current context of characters in a batch
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size] # cropping idx to last block_size token
            logits, loss = self(idx_cond) # getting predictions
            logits = logits[:, -1, :] # focus only on last character (due to its being a bigram model), converting into (B,C)
            probs = F.softmax(logits, dim=-1) # softmaxing probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sampling from distributin, (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # appending sampled index to running sequence, (B, T+1)
        return idx

m = BigramLM()
m.to(device)
xb, yb = get_batch('train')
logits, loss = m(xb, yb)

# training model
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for steps in range(max_iters):
    # evaluate the loss on training and val sets
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
    
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(m.generate(idx=torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))
