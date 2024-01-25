import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Pytorch-specific
    Any instance of a class derived from nn.Module can be called as a function,
    automatically invoking the forward method of the class.
    """

    def __init__(self, vocab_size: int, *args, **kwargs):
        """ Initialise class.
        :param vocab_size: Size of the code book
        """
        super().__init__(*args, **kwargs)
        # Light wrapper around a tensor, initialised with shape(vocab_size, vocab_size)
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, indices, targets):
        """Forward pass on the model

        Definition of logits:
        ----------------------
        I think what it means is, given an index in a batch, give a prediction for how
        likely each of the characters from the code book is likely to follow.
        So the likelihood of character with integer 14 following the character defined in the
        'i'th element of logits, for the 'ib'th batch is logits[ib, i, 14]

        CONFIRM: Channels will be the size of the codebook i.e. vocab_size.

        :param indices: Input integer characters. Tensor of shape (batch, time)
        :param targets: Target integer characters. Tensor of shape (batch, time)
        :return:
        """
        logits = self.token_embedding_table(indices)

        # torch cross_entropy function wants channels as the final dimension (at least when k=1),
        # so reshape by collapsing batch and time dimensions. Note, view won't reallocate memory,
        # assuming type does not change.
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        # Probability of predicting the next characters, plus the target next-characters
        # allows one to definition a loss function
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, indices: torch.tensor, max_new_tokens: int):
        """
        For all batches, extend the time tokens by one per iteration,
        for max_new_tokens iterations
        :param max_new_tokens:
        :return:
        """
        for _ in range(max_new_tokens):
            # Get predictions
            # No reshaping, as want to access a single T (could try this with indexing)
            # Andrej calls self here (forward method), but that's pointless
            logits = self.token_embedding_table(indices)
            # take last time step
            logits = logits[:, -1, :]
            # Get probabilities, applying to all channels of a given batch
            probs = F.softmax(logits, dim=-1)  # Still (B, C)
            # Sample from distribution of probabilities
            next_indices = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running seqeuence
            indices = torch.cat((indices, next_indices), dim=1)  # (B, T+1)
        return indices
