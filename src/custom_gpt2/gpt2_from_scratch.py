"""Generate GPT2 from Scratch
Following this [reference](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=638s)
Also see the paper "Attention is All You Need"

TODOs
* Check out [SentencePiece](https://github.com/google/sentencepiece by Google, and
  [tiktoken](https://github.com/openai/tiktoken) by openAI
"""
from pathlib import Path

import torch
from encoders import char_encoder
from requests_func import request_data

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
    batch_size = 4

    # Some notes
    # Slightly weird way of doing the indexing for me
    # x = training_data[:block_size]
    # y = training_data[1:block_size+1]
    # # Increases the window (context) size from 1 to block_size
    # # x gets increasingly larger, and is used to train the target y
    # for i in range(block_size):
    #     print(x[:i+1], y[i])

    # x, y = get_batch(training_data, block_size, batch_size)

    # At 22.44
