import collections
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import argparse

import torch

import contextlib
import pickle
from supernet_character_llm import BigramLanguageModel
# Encoder: take a string, output a list of integers
def encode(s):
    return [stoi[c] for c in s]

# Decoder: take a list of integers, output a string
def decode(l):
    return ''.join([itos[i] for i in l])  
global data, train_data, valid_data
with open('/work/dlclarge2/sukthank-nanogpt/tutorial/nanoGPT/input.txt', 'r', encoding='utf-8') as f:
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

class NASOptimizer(object):
    """
    Base class for NASBench-101 optimizers. All subclasses should
    inherit from this.
    """

    def __init__(self):
        # get the configuration space
        # configuration (architecture) at each point in time.
        # incumbent_trajectory_error keeps track of the
        # corresponding validation errors of incumbent_trajectory
        self.incumbent_trajectory = []
        self.incumbent_trajectory_error = []
        self.curr_wallclock = 0
        self.curr_incumbent = None
        self.curr_incumbent_error = 10000000
        self.eval_iters = 500

    def optimize(self, n_iters: int = 100):
        raise NotImplementedError

    def sample_random_config(self):
        """
        Return a randomly sampled configuration.
        """
        # TODO: return one randomly sampled configuration from self.cs
        n_layers = [2, 4, 8, 16]
        embed_dims = [32, 64, 128, 256]
        heads = [2, 4, 8, 16]
        activation_indices = [0, 1, 2, 3]
        bias_proj_list = [True, False]
        bias_head_list = [True, False]
        ffwd_bias_list = [True, False]
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
        return config
    
    @torch.no_grad()
    def estimate_loss(self,model,config):
        out = {}
        model.eval()
        for split in ['valid']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, config, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out['valid']

    def train_and_eval(self, config):
        """
        Function that queries the validation error of config
        in self.benchmark. Since every architecture has
        already been trained and evaluated, we just do table
        look-ups without the need to train the neural net.
        """
        model = BigramLanguageModel().cuda()
        model.load_state_dict(torch.load("/work/dlclarge2/sukthank-nanogpt/nanoGPT/model_ofa.pth"))
        valid_accuracy = self.estimate_loss(model,config)
        # TODO: check if config is better than current incumbent
        if min(self.curr_incumbent_error, valid_accuracy) == valid_accuracy:
            self.curr_incumbent_error = valid_accuracy
            self.curr_incumbent = config
            self.incumbent_trajectory.append(config)
            self.incumbent_trajectory_error.append(valid_accuracy)
        else:
            self.incumbent_trajectory.append(self.curr_incumbent)
            self.incumbent_trajectory_error.append(self.incumbent_trajectory_error[-1])
        print("Current incumbent error: ", self.curr_incumbent_error)
        print("Current incumbent: ", self.curr_incumbent)
        with open("incumbent_trajectory_error.pkl", "wb") as f:
            pickle.dump(self.incumbent_trajectory_error, f)
        with open("incumbent_trajectory.pkl", "wb") as f:
            pickle.dump(self.incumbent_trajectory, f)
                    
class RandomSearch(NASOptimizer):
    """
    Algorithm for random search.
    """

    def __init__(self):
        super(RandomSearch, self).__init__()

    def optimize(self, n_iters: int = 100):
        """
        Run random search for n_iters function evaluations.
        """
        for i in range(n_iters):
            config = self.sample_random_config()
            self.train_and_eval(config)

rs = RandomSearch()
rs.optimize(100)