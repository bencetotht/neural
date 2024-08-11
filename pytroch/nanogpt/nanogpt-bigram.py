import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 32 # number of inputs processed in parallel
block_size = 8 # maximum context length
max_iters = 1000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

xb, yb = get_batch('train')

class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C) tensor (batch, time, channel)

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
            logits, loss = self(idx) # getting predictions
            logits = logits[:, -1, :] # focus only on last character (due to its being a bigram model), converting into (B,C)
            probs = F.softmax(logits, dim=-1) # softmaxing probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sampling from distributin, (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # appending sampled index to running sequence, (B, T+1)
        return idx

m = BigramLM(vocab_size)
m.to(device)
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
