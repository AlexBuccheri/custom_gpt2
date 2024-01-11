"""Generate GPT2 from Scratch
Following this [reference](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=638s)
Also see the paper "Attention is All You Need"

TODOs
* Check out [SentencePiece](https://github.com/google/sentencepiece by Google, and
  [tiktoken](https://github.com/openai/tiktoken) by openAI
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from encoders import char_encoder, char_decoder
from requests_func import request_data
from batch import get_batch


class BigramLanguageModel(nn.Module):
    """ Pytorch-specific
    Any instance of a class derived from nn.Module can be called as a function,
    automatically invoking the forward method of the class.
    """

    def __init__(self, vocab_size: int, *args, **kwargs):
        """

        :param vocab_size: Size of the code book
        """
        super().__init__(*args, **kwargs)
        # Light wrapper around a tensor, initialised with shape(vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, indices, targets):
        """ Forward pass on the model
        Both arguments are (batch, time) tensors of shape
        batch_size (or number of batches) and block_size

        :param indices:
        :param targets:
        :return:
        """
        # Given index picks out the the ith row of the embedding table
        # He really does not explain how this then has dimension (B, T, C)
        # I think what it means is, given a index in a batch, give a prediction for how
        # likely each of the characters from the code book is likely to follow. Hence the dimension
        # So likelihood of character with integer 15 following character defined in the ith element of logits, for the
        # ib th batch ,is logits[ib, i, 14]
        logits = self.token_embedding_table(indices)

        # cross_entropy wants channels as the final dimension (at least when k=1), so reshape by collapsing
        # batch and time dimensions. Note, view won't reallocate memory : ) assuming type does not change
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        # Have the probability of predicting the next character, and the next character (the target)
        # so can define the loss function
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
            logits = self.token_embedding_table(indices)
            # take last time step
            logits = logits[:, -1, :]
            # Get probabilities, applying to all channels of a given batch
            probs = F.softmax(logits, dim=-1) # Still (B, C)
            # Sample from distribution of probabilities
            next_indices = torch.multinomial(probs, num_samples=1) # (B, 1)
            indices = torch.cat((indices, next_indices), dim=1) # (B, T+1)
        return indices



if __name__ == "__main__":
    # Get data to learn on
    data_file = Path("data/tiny_shakespeare.txt")

    if not data_file.is_file():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        shakespeare_text = request_data(url)
        with open(data_file, "w") as fid:
            fid.write(shakespeare_text)

    with open(data_file, "r") as fid:
        shakespeare_text = fid.read()

    # All unique characters in text
    chars = sorted(list(set(shakespeare_text)))
    vocab_size = len(chars)

    assert (
        "".join(s for s in chars)
        == "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )
    assert vocab_size == 65

    data = torch.tensor(
        char_encoder(shakespeare_text, chars), dtype=torch.long
    )
    assert data.dtype == torch.int64
    # Could also do len(data) to get total size
    assert list(data.shape) == [1115394]

    n_training = int(0.9 * len(data))
    training_data = data[:n_training]
    validation_data = data[n_training:]

    # Training for LLMs needs block_size + 1, as we're training to make predictions on the block_size,
    # but a prediction requires the ith + 1 character
    block_size = 8
    batch_size = 32

    # Initialise the model
    model = BigramLanguageModel(vocab_size)

    # Can get away with learning rates of > 1.e-4 for small models
    # Takes the gradients and updates the parameters based on these
    optimiser = torch.optim.AdamW(model.parameters(), lr=1.e-3)

    for step in range(10000):
        # Sample a batch of data
        xb, yb = get_batch(training_data, block_size, batch_size)
        # Evaluate the loss
        logits, loss = model(xb, yb)
        # Why does one do this?
        optimiser.zero_grad(set_to_none=True)
        # Gets gradients for all the parameters
        loss.backward()
        # Use gradients to update parameters
        optimiser.step()

    print(loss.item())

    # Define a starting point in the text to predict following characters
    # This is the newline character
    first_integer = torch.zeros((1, 1), dtype=torch.long)
    prediction = model.generate(first_integer, max_new_tokens=300)[0].tolist()
    print(char_decoder(prediction, chars))

    # At time 37:48
