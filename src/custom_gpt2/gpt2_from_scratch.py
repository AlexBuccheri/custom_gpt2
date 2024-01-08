"""Generate GPT2 from Scratch
Following this [reference](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=638s)

TODOs
* Check out [SentencePiece](https://github.com/google/sentencepiece by Google, and
  [tiktoken](https://github.com/openai/tiktoken) by openAI
"""
from pathlib import Path

from encoders import char_decoder, char_encoder
from requests_func import request_data

if __name__ == "__main__":
    # Get data to learn on
    data_file = Path("data/tiny_shakespeare.txt")

    if not data_file.is_file():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data = request_data(url)
        with open(data_file, "w") as fid:
            fid.write(data)

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
