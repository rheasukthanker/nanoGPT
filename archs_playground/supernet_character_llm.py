import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
with open('/work/dlclarge2/sukthank-nanogpt/nanoGPT/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
# data loading
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 64  # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
dropout = 0.0

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(config):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, config, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(input)


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, block_size, n_embd, dropout, head_size, bias_proj=False, bias_head=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, block_size, dropout, head_size, bias=bias_head)
                                   for _ in range(num_heads)])  # Slightly more efficient version below
        self.proj = nn.Linear(n_embd, n_embd, bias=bias_proj)
        self.dropout = nn.Dropout(dropout)

    def sample_proj(self, i, choices):
        embed_dim = choices["embed_dim"]
        bias_proj = choices["bias_proj"][i]
        if bias_proj:
            return self.proj.weight[:embed_dim, :embed_dim], self.proj.bias[:embed_dim]
        else:
            return self.proj.weight[:embed_dim, :embed_dim], None

    def forward(self, x, i, choices):
        # Simply stack multiple heads
        out = torch.cat([h(x, i, choices) for h in self.heads], dim=-1)
        weight, bias = self.sample_proj(i, choices)
        out = self.dropout(torch.nn.functional.linear(out[:,:,:choices["embed_dim"]], weight, bias))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout, activation_id=0, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )
        self.activation_id = activation_id
        self.activations = [nn.ReLU(), nn.GELU(), SiLUActivation(), new_gelu]

    def get_weights(self, i, choices):
        bias_net_0 = choices["net_0_bias"][i]
        bias_net_1 = choices["net_1_bias"][i]
        n_embd = choices["embed_dim"]
        if bias_net_0==True and bias_net_1==True:
           return self.net[0].weight[:4*n_embd,:n_embd], self.net[0].bias[:4*n_embd], self.net[1].weight[:n_embd,:4*n_embd], self.net[1].bias[:n_embd]
        elif bias_net_0==True and bias_net_1==False:
              return self.net[0].weight[:4*n_embd,:n_embd], self.net[0].bias[:4*n_embd], self.net[1].weight[:n_embd,:4*n_embd], None
        elif bias_net_0==False and bias_net_1==True:
              return self.net[0].weight[:4*n_embd,:n_embd], None, self.net[1].weight[:n_embd,:4*n_embd], self.net[1].bias[:n_embd]
        else:
              return self.net[0].weight[:4*n_embd,:n_embd], None, self.net[1].weight[:n_embd,:4*n_embd], None

    def forward(self, x, i, choices):
        activation_id = choices["activation_id"][i]
        weight_0, bias_0, weight_1, bias_1 = self.get_weights(i, choices)
        x = torch.nn.functional.linear(x[:,:,:choices["embed_dim"]], weight_0, bias_0)
        x = self.activations[activation_id](x)
        x = torch.nn.functional.linear(x[:,:,:4*choices["embed_dim"]], weight_1, bias_1)
        x = self.net[2](x)
        return x


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, block_size, dropout, head_size, bias=False):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=bias)
        self.query = nn.Linear(n_embd, head_size, bias=bias)
        self.value = nn.Linear(n_embd, head_size, bias=bias)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def get_bias(self,bias_choice, head_size):
        if bias_choice== 'True':
            bias = self.key.bias[:head_size]
        else:
            bias = None
        return bias

    def sample_kqv(self, i, choices):
        head_size = choices['head_size'][i]
        embed_dim = choices['embed_dim']
        bias_k = self.get_bias(choices['bias_head_k'][i], head_size)
        bias_q = self.get_bias(choices['bias_head_q'][i], head_size)
        bias_v = self.get_bias(choices['bias_head_v'][i], head_size)
        return self.key.weight[:head_size,:embed_dim], self.query.weight[:head_size,:embed_dim], self.value.weight[:head_size,:embed_dim], bias_k, bias_q, bias_v


    def forward(self, x, i, choices):
        B, T, C = x.shape
        kw , qw, vw, bias_k, bias_q, bias_v = self.sample_kqv(i, choices)
        embed_dim = choices['embed_dim']
        k = torch.nn.functional.linear(x[:,:,:embed_dim], kw, bias_k)
        q = torch.nn.functional.linear(x[:,:,:embed_dim], qw, bias_q)
        v  = torch.nn.functional.linear(x[:,:,:embed_dim], vw, bias_v)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, num_heads, block_size, head_size, dropout, bias_proj=False, bias_head=False, ffwd_bias=False, activation_id=0):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(
            num_heads, block_size, n_embd, dropout, head_size, bias_proj=bias_proj, bias_head=bias_head)
        self.ffwd = FeedFoward(
            n_embd, dropout, activation_id=activation_id, bias=ffwd_bias)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def get_weights_layernorm(self, choices):
        embed_choice = choices['embed_dim']
        return self.ln1.weight[:embed_choice], self.ln1.bias[:embed_choice], self.ln2.weight[:embed_choice], self.ln2.bias[:embed_choice]

    def forward(self, x, i, choices):
        ln1_weight, ln1_bias, ln2_weight, ln2_bias = self.get_weights_layernorm(choices)
        x = x + self.sa(torch.nn.functional.layer_norm(x,[choices["embed_dim"]],weight=ln1_weight, bias=ln1_bias), i, choices)
        x = x + self.ffwd(torch.nn.functional.layer_norm(x,[choices["embed_dim"]],weight=ln2_weight, bias=ln2_bias), i, choices)
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self, n_layers=16, vocab_size=len(chars), n_embd=256, block_size=64, num_heads=16, head_size=128, dropout=0.0, bias_proj=True, bias_head=True, ffwd_bias=True, activation_id=0):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # Note attention does not have any notion of colocation of characters/words and this is important for lms
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads, block_size, head_size, dropout, bias_proj=bias_proj,
                                    bias_head=bias_head, ffwd_bias=ffwd_bias, activation_id=activation_id) for i in range(self.n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True)

    def get_weights_layernorm(self, choices):
        embed_choice = choices['embed_dim']
        return self.ln_f.weight[:embed_choice], self.ln_f.bias[:embed_choice]

    def get_weights_embedding(self, choices):
        embed_choice = choices['embed_dim']
        return self.token_embedding_table.weight[:,:embed_choice], self.position_embedding_table.weight[:,:embed_choice]

    def get_weights_lm_head(self, choices):
        embed_choice = choices['embed_dim']
        bias_lm_head = choices["bias_lm_head"]
        if bias_lm_head:
            return self.lm_head.weight[:,:embed_choice], self.lm_head.bias[:]
        else:
            return self.lm_head.weight[:,:embed_choice], None

    def forward(self, idx, choices, targets=None):
        B, T = idx.shape
        #print(choices)
        token_embedding_weight, position_embedding_weight = self.get_weights_embedding(choices)
        ln_f_weight, ln_f_bias = self.get_weights_layernorm(choices)
        lm_head_weight, lm_head_bias = self.get_weights_lm_head(choices)
        # idx and targets are both (B,T) tensor of integers
        tok_emb = torch.nn.functional.embedding(idx, token_embedding_weight)  # (B,T,C)
        pos_emb =torch.nn.functional.embedding(torch.arange(T, device=device), position_embedding_weight)  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        for i in range(choices["n_layer"]):
            x = self.blocks[i](x, i, choices)
        x = torch.nn.functional.layer_norm(x, [choices["embed_dim"]], ln_f_weight, ln_f_bias)  # (B,T,C)
        logits = torch.nn.functional.linear(x[:,:,:choices["embed_dim"]],lm_head_weight, lm_head_bias)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


'''if __name__ == '__main__':
    batch_size = 64  # how many independent sequences will we process in parallel?
    block_size = 64  # what is the maximum context length for predictions?
    max_iters = 1000
    eval_interval = 100
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    dropout = 0.0
    # ------------
    n_layers = [2, 4, 8, 16]
    embed_dims = [32, 64, 128, 256]
    heads = [2, 4, 8, 16]
    activation_indices = [0, 1, 2, 3]
    bias_proj_list = [True, False]
    bias_head_list = [True, False]
    ffwd_bias_list = [True, False]
    #model = torch.nn.DataParallel(BigramLanguageModel())
    model = BigramLanguageModel()
    model = model.to(device)
    stats = {}
    for _ in range(1000000):
        torch.manual_seed(1337)
        stats_now = {}
        config = {}
        n_layer = np.random.choice(n_layers)
        config["n_layer"] = n_layer
        n_embd = np.random.choice(embed_dims)
        config["embed_dim"] = n_embd
        n_head = [np.random.choice(heads) for _ in range(n_layer)]
        config["n_head"] = n_head
        activation_id = [np.random.choice(
            activation_indices) for _ in range(n_layer)]
        config["activation_id"] = activation_id
        bias_proj = [np.random.choice(bias_proj_list) for _ in range(n_layer)]
        config["bias_proj"] = bias_proj
        bias_head_q = [np.random.choice(bias_head_list) for _ in range(n_layer)]
        config["bias_head_q"] = bias_head_q
        bias_head_k = [np.random.choice(bias_head_list) for _ in range(n_layer)]
        config["bias_head_k"] = bias_head_k
        bias_head_v = [np.random.choice(bias_head_list) for _ in range(n_layer)]
        config["bias_head_v"] = bias_head_v
        net_0_bias = [np.random.choice(ffwd_bias_list) for _ in range(n_layer)]
        config["net_0_bias"] = net_0_bias
        net_1_bias = [np.random.choice(ffwd_bias_list) for _ in range(n_layer)]
        config["net_1_bias"] = net_1_bias
        lm_head_bias = np.random.choice([True, False])
        config["bias_lm_head"] = lm_head_bias
        head_size = [n_embd//h for h in n_head]
        config["head_size"] = head_size

        # print the number of parameters in the model
        print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
        num_params = sum(p.numel() for p in model.parameters())/1e6
        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max_iters)
        val_losses = []
        str_n_head = '_'.join([str(h) for h in n_head])
        str_activation_id = '_'.join([str(a) for a in activation_id])
        str_bias_proj = '_'.join([str(b) for b in bias_proj])
        str_bias_head_q = '_'.join([str(b) for b in bias_head_q])
        str_bias_head_k = '_'.join([str(b) for b in bias_head_k])
        str_bias_head_v = '_'.join([str(b) for b in bias_head_v])
        str_net_0_bias = '_'.join([str(b) for b in net_0_bias])
        str_net_1_bias = '_'.join([str(b) for b in net_1_bias])
        if ' '.join([str(n_layer), str(n_embd), str_n_head, str_activation_id, str_bias_proj, str_bias_head_q, str_bias_head_k, str_bias_head_v, str_net_0_bias, str_net_1_bias, str(num_params)]) not in stats:
            for iter in range(max_iters):
                # every once in a while evaluate the loss on train and val sets
                if iter % eval_interval == 0 or iter == max_iters - 1:
                    losses = estimate_loss(config)
                    print(
                        f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    val_losses.append(losses['val'].item())
                # sample a batch of data
                xb, yb = get_batch('train')
                # evaluate the loss
                logits, loss = model(xb, config, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
            # stats[' '.join([str(n_layer), str(n_embd), str(n_head), str(activation_id), str(bias_proj), str(bias_head), str(ffwd_bias), str(num_params)])] = val_losses
            stats_now["val_losses"] = val_losses
            stats_now["n_layer"] = n_layer
            stats_now["n_embd"] = n_embd
            stats_now["n_head"] = n_head
            stats_now["activation_id"] = activation_id
            stats_now["bias_proj"] = bias_proj
            stats_now["bias_head_q"] = bias_head_q
            stats_now["bias_head_k"] = bias_head_k
            stats_now["bias_head_v"] = bias_head_v
            stats_now["net_0_bias"] = net_0_bias
            stats_now["net_1_bias"] = net_1_bias
            stats_now["num_params"] = num_params
            stats[' '.join([str(n_layer), str(n_embd), str_n_head, str_activation_id,
                           str_bias_proj, str_bias_head_q, str_bias_head_k, str_bias_head_v,  str_net_0_bias, str_net_1_bias, str(num_params)])] = stats_now
            with open('stats_ofa.pkl', 'wb') as fp:
                pickle.dump(stats, fp)
            torch.save(model.state_dict(), 'model_ofa.pth')'''
