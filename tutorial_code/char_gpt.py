import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

import tiktoken
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
warmup_iters = 100
lr_decay_iters = 1000
min_lr = 6e-5
learning_rate = 6e-4
decay_lr = True
def to_tokens(example):
    """Function to tokenize a string using BPE.
    """
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(example["text"]) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

# Encoder: take a string, output a list of integers
def encode(s):
    return [stoi[c] for c in s]

# Decoder: take a list of integers, output a string
def decode(l):
    return ''.join([itos[i] for i in l])


def plot_losses(losses, verbosity, val_losses=None):
    # plt.clf()
    plt.plot(losses, label="train");
    if val_losses is not None:
        plt.plot(val_losses, label="valid");
        plt.legend();
    plt.ylabel("Loss")
    plt.xlabel(f"Num steps (~{verbosity}x)")
    plt.xlim(0, len(losses))
    plt.show();

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type, weight_tying):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        #print(no_decay)
        #print(decay)
        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        if weight_tying:
           decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        #print(len(param_dict.keys() - union_params))
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer
    
def get_batch( split: str, block_size: int = 8, batch_size: int = 4, device: str = None):
    """ Gets a randomized batch from the split of data chosen.

    Arguments
    ---------
    split : str, {"train", "valid"}
    block_size : int
        The context length for predictions, that is, a sentence length
    batch_size : int
        The batch size, that is, the number of sentences
    """
    # generate a small batch of data of inputs x and targets y
    assert split in ["train", "valid"]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = train_data if split == 'train' else valid_data
    # generating random indices as markers in the full text document
    # such that they are a starting point to the sentence of length
    # `block_size` that will be a data point in the batch
    ix = torch.randint(
        low=0, high=len(data) - block_size, size=(batch_size,)
    )
    # extracting a sentence of length `block_size` for every
    # random starting point in `ix`
    x = torch.stack([data[i:i+block_size] for i in ix])
    # extracting a sentence of length `block_size` for every
    # random starting point in `ix` + 1 (shifted to right)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(train_data, valid_data, model: nn.Module, eval_iters: int):
    """ Function to evaluate the model on train & valid splits.
    """
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_and_evaluate_model(
    model: nn.Module,
    block_size: int,
    batch_size: int,
    optimizer: torch.optim = None,
    num_train_steps: int = 10000,
    verbosity_len: int = 1000,
    eval_iters: int = 500,
    plot_loss: str = True,
    device: str = "cpu",
    **kwargs
):
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=kwargs["learning_rate"]
        )

    train_losses = [np.inf]
    valid_losses = [np.inf]

    for iter in tqdm(range(num_train_steps)):

        # sample a batch of data
        xb, yb = get_batch('train', block_size, batch_size, device)

        # evaluate loss on the batch
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)

        # gradient update
        loss.backward()
        optimizer.step()

        # every once in a while evaluate the loss on train and val sets
        if iter % verbosity_len == 0 or iter == num_train_steps - 1:
            _losses = estimate_loss(train_data, valid_data, model, eval_iters)
            train_losses.append(_losses['train'])
            valid_losses.append(_losses['valid'])
            print()
            print(
                f"step {iter}: train loss {_losses['train']:.4f}, "\
                f"val loss {_losses['valid']:.4f}"
            )

    if plot_loss:
        plot_losses(train_losses, verbosity_len, valid_losses)



def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def train_and_evaluate_nanoGPT(
    model: nn.Module,
    block_size: int,
    batch_size: int,
    optimizer: torch.optim = None,
    num_train_steps: int = 10000,
    verbosity_len: int = 1000,
    eval_iters: int = 500,
    plot_loss: str = True,
    device: str = "cpu",
    **kwargs
):
    model.train()
    if optimizer is None:
        optimizer = configure_optimizers(model,weight_decay=0.01,learning_rate=6e-4,betas=(0.9,0.95),device_type=device, weight_tying=model.weight_tying)
        #torch.optim.AdamW(
        #    model.parameters(), lr=kwargs["learning_rate"]
        #)

    train_losses = [np.inf]
    valid_losses = [np.inf]

    for iter in tqdm(range(num_train_steps)):
        lr = get_lr(iter) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # sample a batch of data
        xb, yb = get_batch('train', block_size, batch_size, device)

        # evaluate loss on the batch
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)

        # gradient update
        loss.backward()
        optimizer.step()

        # every once in a while evaluate the loss on train and val sets
        if iter % verbosity_len == 0 or iter == num_train_steps - 1:
            _losses = estimate_loss(train_data, valid_data, model, eval_iters)
            train_losses.append(_losses['train'])
            valid_losses.append(_losses['valid'])
            print()
            print(
                f"step {iter}: train loss {_losses['train']:.4f}, "\
                f"val loss {_losses['valid']:.4f}"
            )

    if plot_loss:
        plot_losses(train_losses, verbosity_len, valid_losses)

def generate_from_model(
    model: nn.Module,
    vocab_size: int,
    block_size: int,
    sentence_len: int,
    batch_num: int = 1,
    start_str: str = None,
    device: str = "cpu"
):
    # sampling a start token and generating a batch of it as context
    if start_str is None:
        start_token = np.random.randint(vocab_size)
        print(f"Start token: {decode([start_token])}")
        context = torch.zeros((batch_num, 1), dtype=torch.long, device=device)
        # setting the first token of the batch to the sampled start token
        context[:, 0] = start_token
    else:
        start_token = encode(start_str)
        print(f"Start token: {decode(start_token)}")
        # generating batch of sentences with the start token
        context = torch.tensor(start_token, dtype=torch.long, device=device)
        context = context.repeat(batch_num, 1)
    # will generate the next sentence_len characters for each of the start token
    out = model.generate(context, max_new_tokens=sentence_len)
    print(out.shape)
    return out


def decode_and_print_batch(batch):
    for b in range(batch.shape[0]):
        print(f"\nBatch ID: {b}")
        print(decode(batch[b].tolist()))
    print()


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, block_size: int, n_embed: int, head_size: int, flash: bool = False):
        """
        Arguments
        ---------
        block_size: int
            The sentence length allowed
        n_embed: int
            The size (dimension) of the token embeddings
        head_size: int
            The size (dimension) of the output of an attention head
        """
        super().__init__()

        self.block_size = block_size  # equivalent to T
        self.n_embed = n_embed
        self.head_size = head_size  # equivalent to C
        self.flash = flash
        self.key = nn.Linear(self.n_embed, self.head_size, bias=False)
        self.query = nn.Linear(self.n_embed, self.head_size, bias=False)
        self.value = nn.Linear(self.n_embed, self.head_size, bias=False)

        self.register_buffer(
            'tril', torch.tril(torch.ones(self.block_size, self.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape  # B: batch size; T: block size; C: embedding size
        k = self.key(x)  # (B, T, C) @ (C, head_size)  -> (B, T, head_size)
        q = self.query(x)  # (B, T, C) @ (C, head_size)  -> (B, T, head_size)
        v = self.value(x)  # (B, T, C) @ (C, head_size)  -> (B, T, head_size)
        if self.flash:
           out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.1, is_causal=True)
        else:
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
            # performing `scaled` attention
            wei *= self.head_size ** -(1 / 2)  # scaling by `1/sqrt(head size)`
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
            wei = F.softmax(wei, dim=-1)  # (B, T, T)
            # perform the weighted aggregation of the values
            out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(
        self, block_size: int, n_embed: int, head_size: int, num_heads: int, flash: bool = False
    ):
        """
        Arguments
        ---------
        block_size: int
            The sentence length allowed
        n_embed: int
            The size (dimension) of the token embeddings
        head_size: int
            The size (dimension) of the output of an attention head
        num_heads: int
            The number of single attention heads that together form
            one multi-headed attention layer
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(block_size, n_embed, head_size, flash=flash) for _ in range(num_heads)]
        )
        # linear FC layer
        self.proj = nn.Linear(head_size * num_heads, n_embed)

    def forward(self, x):
        # simply stack multiple heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # B: batch size; T: block size; C: embedding size; H: head_size * num_heads
        out = self.proj(out)  # (B, T, H) @ (H, C) -> (B, T, C)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(
            self,
            n_embed: int,
            wide_factor: int = 4,
            activation: str = "relu",
            dropout: float = 0.0
        ):
        super().__init__()
        self.activation = nn.ReLU if activation == "relu" else nn.GELU
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(n_embed, wide_factor * n_embed),
            self.activation(),
            nn.Linear(wide_factor * n_embed, n_embed),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(
        self,
        block_size: int,
        n_embed: int,
        num_heads: int,
        wide_factor: int = 4,
        activation: str = "relu",  # could also be "gelu"
        dropout: float = 0.0,
        flash: bool = False,
        prenormalize: bool = False
    ):
        super().__init__()
        # setting head_size to be a factor of other dimensions
        head_size = n_embed // num_heads
        # the multi-headed self-attention (msa)
        self.msa = MultiHeadAttention(block_size, n_embed, head_size, num_heads, flash=flash)
        self.ffwd = FeedForward(n_embed, wide_factor, activation, dropout)

        self.prenormalize = prenormalize
        if prenormalize:
            self.pre_ln = nn.LayerNorm(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        if self.prenormalize:
            # normalizes inputs before passing it through the attention block
            x = x + self.msa( self.pre_ln(x) )
        else:
            x = x + self.msa(x)
        # norm after attention
        x = self.ln1(x)
        # feed-forward
        x = x + self.ffwd(x)
        # norm after feed-forward
        x = self.ln2(x)
        return x


class CharGPT(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            n_layers: int,
            block_size: int,
            n_embed: int,
            num_heads: int,
            wide_factor: int = 4,
            flash: bool = False,
            activation: str = "relu",  # could also be "gelu"
            dropout: float = 0.0,
            prenormalize: bool = False,
            device: str = None
    ):
        super().__init__()
        # each token directly reads off the logits for the next
        # token from a lookup table
        # Note attention does not have any notion of colocation
        # of characters/words and this is important for lms
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(
                block_size=block_size,
                n_embed=n_embed,
                num_heads=num_heads,
                wide_factor=wide_factor,
                activation=activation,
                dropout=dropout,
                flash = flash,
                prenormalize=prenormalize,
            ) for _ in range(n_layers)]  # stacks the layers of Transformer blocks
        )
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm (has bias)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.block_size = block_size
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.activation = activation
        self.device = device
        if self.device is None:
            self.device = "gpu" if torch.cuda.is_available() else "cpu"

    def forward(self, idx, targets=None):
        # B: batch_size, T: block_size, C: embedding_size
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        # fixing positional inputs and learning an embedding over it
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        # adding the positional embeddings across the token embedding batch
        x = tok_emb + pos_emb  # (B,T,C)
        # forward pass through the Transformer layers
        x = self.blocks(x)  # (B,T,C)
        # final layernorm
        x = self.ln_f(x)  # (B,T,C)
        # projecting to the vocabulary
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # B: batch_size, T: block_size, C:
        # idx is (B, T) array of indices in the current context

        self.eval()

        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        self.train()
        return idx


class NanoGPT(CharGPT):
    def __init__(
            self,
            vocab_size: int,
            n_layers: int,
            block_size: int,
            n_embed: int,
            num_heads: int,
            wide_factor: int = 4,
            activation: str = "relu",  # could also be "gelu"
            dropout: float = 0.0,
            prenormalize: bool = False,
            flash: bool = False,
            device: str = None,
            weight_tying: str = False,
            init_type: str = None
    ):
        super().__init__(
            vocab_size=vocab_size,
            n_layers=n_layers,
            block_size=block_size,
            n_embed=n_embed,
            num_heads=num_heads,
            wide_factor=wide_factor,
            activation=activation,
            dropout=dropout,
            flash = flash,
            prenormalize=prenormalize,
            device=device,
        )
        self.blocks = nn.Sequential(
            nn.Dropout(dropout),  # like GPT-2
            *[Block(
                block_size=block_size,
                n_embed=n_embed,
                num_heads=num_heads,
                wide_factor=wide_factor,
                activation=activation,
                dropout=dropout,
                flash = flash,
                prenormalize=prenormalize,
            ) for _ in range(n_layers)]  # stacks the layers of Transformer blocks
        )
        self.weight_tying = weight_tying
        if weight_tying:
            # weight-tying, https://paperswithcode.com/method/weight-tying
            self.token_embedding_table.weight = self.lm_head.weight

        if init_type == "custom":
            # https://github.com/karpathy/nanoGPT/blob/master/model.py#L147
            # init all weights
            self.apply(self._init_weights)
            # apply special scaled init to the residual projections, per GPT-2 paper
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding_table.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


global data, train_data, valid_data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Checking all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
vocab_set = "".join(chars)

# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Train and test splits
train_size = 0.9
data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_size * len(data))
train_data = data[:n]
valid_data = data[n:]
'''model = CharGPT(
    vocab_size,
    3,
    block_size=8,
    n_embed=32,
    num_heads=2,
    wide_factor=4,
    activation="relu",
    dropout=0.1,
    flash=True,
    prenormalize=False,
    device="cpu",
)
train_and_evaluate_model(model,8,2,learning_rate=0.001 )'''
'''BLOCK_SIZE = 64
EMBED_SIZE = 128
NUM_HEADS = 6
ACTIVATION = "gelu"
PRENORM = True
WIDE_FACTOR = 4
DROPOUT = 0.0
LAYERS = 6
WEIGHT_TYING = False
FLASH = True
INIT_TYPE = "custom"

# Training HPs
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Experiment HPs
VOCAB_SIZE = 50304
NUM_TRAIN_STEPS = 10000
VERBOSTIY_LEN = 1000
EVAL_ITERS = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = NanoGPT(
    vocab_size=VOCAB_SIZE,
    n_layers=LAYERS,
    block_size=BLOCK_SIZE,
    n_embed=EMBED_SIZE,
    num_heads=NUM_HEADS,
    wide_factor=WIDE_FACTOR,
    activation=ACTIVATION,
    dropout=DROPOUT,
    prenormalize=PRENORM,
    flash = FLASH,
    device=DEVICE,
    weight_tying=WEIGHT_TYING,
    init_type=INIT_TYPE,
)
model = model.to(DEVICE)
train_and_evaluate_nanoGPT(
    model=model,
    block_size=BLOCK_SIZE,
    batch_size=BATCH_SIZE,
    optimizer=None,  # internally uses AdamW, pass an optimizer to override
    num_train_steps=NUM_TRAIN_STEPS, 
    verbosity_len=VERBOSTIY_LEN, 
    eval_iters=EVAL_ITERS,
    plot_loss=True, 
    device=DEVICE,
    learning_rate=LEARNING_RATE  # part of kwargs
);'''