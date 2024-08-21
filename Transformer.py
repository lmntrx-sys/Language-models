from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from bitnet import BitLinear

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 500
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
n_embd = 384
n_head = 6
n_layer = 6
vocab_size = 64
num_groups = 5
dropout = 0.2
# ------------

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int):
        super().__init__()
        self.key = BitLinear(n_embd, head_size, bias=False, num_groups=num_groups)
        self.query = BitLinear(n_embd, head_size, bias=False, num_groups=num_groups)
        self.value = BitLinear(n_embd, head_size, bias=False, num_groups=num_groups)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Head module in the Transformer architecture.
        
        This function takes in an input tensor `x` of shape (batch, time-step, channels) and 
        outputs a tensor of shape (batch, time-step, head size) after applying self-attention.
        
        Parameters:
        x (Tensor): Input tensor of shape (batch, time-step, channels)
        
        Returns:
        Tensor: Output tensor of shape (batch, time-step, head size)
        """
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = BitLinear(head_size * num_heads, n_embd, num_groups=num_groups)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MultiHeadAttention module in the Transformer architecture.

        This function takes in an input tensor `x` and outputs a tensor after applying multi-head self-attention.

        Parameters:
        x (Tensor): Input tensor

        Returns:
        Tensor: Output tensor after applying multi-head self-attention
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            BitLinear(n_embd, 4 * n_embd, num_groups=num_groups),
            nn.GELU(),
            BitLinear(4 * n_embd, n_embd, num_groups=num_groups),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the FeedFoward module in the Transformer architecture.

        This function takes in an input tensor `x` and outputs a tensor after applying a linear layer followed by a non-linearity.

        Parameters:
        x (Tensor): Input tensor

        Returns:
        Tensor: Output tensor after applying a linear layer followed by a non-linearity
        """
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd: int, n_head: int):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Transformer block.

        This function takes in an input tensor `x` and outputs a tensor after applying multi-head self-attention and a feed-forward network.

        Parameters:
        x (Tensor): Input tensor

        Returns:
        Tensor: Output tensor after applying multi-head self-attention and a feed-forward network
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = BitLinear(n_embd, vocab_size, num_groups=num_groups)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, BitLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets=None) -> Tuple[torch.Tensor, float]:
        """
        Defines the forward pass of the GPT language model.

        This function takes in an input tensor `idx` and an optional target tensor `targets`. It returns a tuple containing the logits and the loss.

        Parameters:
        idx (Tensor): Input tensor of shape (B, T) containing integer values representing the input sequence.
        targets (Tensor, optional): Target tensor of shape (B, T) containing integer values representing the target sequence. Defaults to None.

        Returns:
        tuple: A tuple containing the logits and the loss.
        The logits are a tensor of shape (B, T, vocab_size) representing the predicted probabilities for each token in the input sequence.
        The loss is a tensor representing the cross-entropy loss between the predicted probabilities and the target sequence.
        """
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        B,T, C = logits.shape

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    	
        #For each block within the batch, this method generates the most probable token(s) to come next
        #For simple use, pass a batch of only one block to this method in order to simulate "ChatGPT" like interaction of "prompt" and "answer"
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx




