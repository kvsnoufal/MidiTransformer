# https://habr.com/en/companies/ods/articles/708672/
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class AttentionHead(nn.Module):
    """
    One head of the self-attention layer
    """

    def __init__(self, head_size, num_embed, block_size):
        super().__init__()
        """
        Initializes the AttentionHead module.

        Args:
            head_size (int): The size of each attention head.
            num_embed (int): The dimension of the input embeddings.
            block_size (int): The block size of the input sequence.
        """
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        # tril is a lower triangular matrix. it is not a parameter
        # of the model, so we assign it to the module using register_buffer
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        Performs the forward pass of the AttentionHead module.

        Args:
            x (Tensor): The input tensor of shape (B, T, C), where B is the batch size, T is the sequence length,
                and C is the dimension of the input embeddings.

        Returns:
            Tensor: The output tensor of shape (B, T, C).
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        # we need to transpose k to match q
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # Tril matrix (lower triagular matrix) is used to mask 
        # future positions (setting them to -inf) so that the
        # decoder "learns" to predict next words
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        # weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B,T,T) @ (B,T,C) ---> (B,T,C)
        return out
class MultiHeadAttention(nn.Module):
    """
    Multiple Heads of self-attention in parallel
    """

    def __init__(self, num_heads, head_size, num_embed, block_size):
        """
        Initializes the MultiHeadAttention module.

        Args:
            num_heads (int): The number of attention heads.
            head_size (int): The size of each attention head.
            num_embed (int): The dimension of the input embeddings.
            block_size (int): The block size of the input sequence.
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    head_size=head_size,
                    num_embed=num_embed,
                    block_size=block_size,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(num_embed, num_embed)

    def forward(self, x):
        # output of the self-attention
       
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
         # (B, T, num_heads * head_size)
        # apply the linear projection layer
        out = self.proj(out)
        # (B, T, num_embed)
        return out    
class FeedForward(nn.Module):
    """
    A simple linear layer followed by ReLu
    """

    def __init__(self, num_embed):
        super().__init__()
        self.net = nn.Sequential(
            # in the Attention is All You Need paper
            # authors are using the size of the ffwd layer 2048
            # and the output of the model is 512
            # so we apply the same factor of 4
            nn.Linear(num_embed, 4 * num_embed),
            nn.ReLU(),
            # apply the linear projection layer
            nn.Linear(4 * num_embed, num_embed),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Groups together MultiHeadAttention and FeedForward modules
      to form a Transformer block.
    """

    def __init__(self, num_heads, block_size, num_embed):
        super().__init__()
        head_size = num_embed // num_heads
        self.sa = MultiHeadAttention(
            num_heads=num_heads,
            head_size=head_size,
            num_embed=num_embed,
            block_size=block_size,
        )
        self.ffwd = FeedForward(num_embed=num_embed)
        # add the layer normalization
        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)

    def forward(self, x):
        # "x +" is the skip (or residual) connection
        # it helps with optimization
        # also we apply layer normalization before self-attention
        # and feed-forward (a reshufle from original paper)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x    
class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # a simple lookup table that stores embeddings of a fixed dictionary and size
        # each token directly reads off the logits for the next token from a lookup table
        # see more: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.vocab_size = kwargs.get("vocab_size", 100)
        self.num_embed = kwargs.get("num_embed", 32)
        self.block_size = kwargs.get("block_size", 8)
        self.num_heads = kwargs.get("num_heads", 4)
        self.num_layers = kwargs.get("num_layers", 4)
        # each token reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.num_embed)
        # each position from 0 to block_size-1 will get its embedding
        self.position_embedding_table = nn.Embedding(self.block_size, self.num_embed)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    num_heads=self.num_heads,
                    block_size=self.block_size,
                    num_embed=self.num_embed,
                )
                for _ in range(self.num_layers)
            ]
        )
        # we add the layer norm before the Linear layer
        self.ln_f = nn.LayerNorm(self.num_embed)
        self.lm_head = nn.Linear(self.num_embed, self.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are (B,T) tensor of integers
        # the token_emb is (B, T, C), C = NUM_EMBED
        token_emb = self.token_embedding_table(idx)
        # (T, C)
        posit_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))

        x = token_emb + posit_emb
        # apply one head of self-attention
        x = self.blocks(x)
        # (B, T, vocab_size)
        logits = self.lm_head(x)
        # compute the loss
        if targets != None:
            # cross_entropy accepts inputs in a (batch_size, num_classes)
            # so we need to reformat our logits dimensions to
            # (batch_size * time, dim_vocabulary), time = block_size
            B, T, C = logits.shape
            logits = torch.reshape(logits, (B * T, C))
            targets = torch.reshape(targets, (B * T,))
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context too the  last block_size tokens
            # because tokens don't communicate between blocks
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution with probabilities probs
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx    