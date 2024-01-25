"""Generate GPT2 from Scratch
Following this [reference](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=638s)
Also see the paper "Attention is All You Need"

TODOs
* Check out [SentencePiece](https://github.com/google/sentencepiece by Google, and
  [tiktoken](https://github.com/openai/tiktoken) by openAI
"""
import dataclasses
from pathlib import Path

import torch
from batch import get_batch
from bigram import BigramLanguageModel
from cuda_settings import device
from encoders import char_decoder, char_encoder
from requests_func import request_data

# Pin seed to get results consistent with the tutorial
torch.manual_seed(1337)


@dataclasses.dataclass
class HyperParameters:
    # Number of independent sequences processed in parallel
    batch_size: int = 32
    # Max number of characters to consider for learning == max context length
    block_size: int = 8
    # Number of learning iterations
    max_iters: int = 3000
    # Learning rate
    # Can get away with learning rates of > 1.e-4 for small models i.e. e-3 to e-2
    learning_rate: int = 1.0e-2
    # Add comment
    eval_interval: int = 300
    # Add comment
    eval_iters: int = 200


# no Grad avoids calling backwards on everything in the function => more efficient memory use
@torch.no_grad()
def estimate_loss(training_data, validation_data, params: HyperParameters, model) -> dict:
    """

    :return:
    """
    estimates = {}
    # Switch the model into evaluation model
    model.eval()
    for label, data in [('train', training_data), ('val', validation_data)]:
        losses = torch.zeros(params.eval_interval)
        for i in range(params.eval_interval):
            x, y = get_batch(data, params.block_size, params.batch_size)
            _, loss = model(x, y)
            losses[i] = loss.item()
        estimates[label] = losses.mean()
    # Switch back to training mode
    model.train()
    return estimates



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

    # Split data into training and validation
    n_training = int(0.9 * len(data))
    training_data = data[:n_training]
    validation_data = data[n_training:]

    # Hyperparameter defaults
    params = HyperParameters()

    # Initialise the model
    model = BigramLanguageModel(vocab_size)
    m = model.to(device)

    # Takes the gradients and updates the parameters based on these
    optimiser = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)

    # Training loop
    for step in range(params.max_iters):

        # Evaluate the loss at set intervals
        if step % params.eval_interval == 0:
            losses = estimate_loss(training_data, validation_data, params, model)
            print(f"Step {step}: Training loss {losses['train']:.4f}. Validation loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch(training_data, params.block_size, params.batch_size)

        # Evaluate the loss
        logits, loss = model(xb, yb)
        # What does one do this?
        optimiser.zero_grad(set_to_none=True)
        # Gets gradients for all the parameters
        loss.backward()
        # Use gradients to update parameters
        optimiser.step()

    # print(loss.item())

    # Define a starting point in the text to predict following characters
    # This is the newline character
    first_integer = torch.zeros((1, 1), dtype=torch.long, device=device)
    prediction = model.generate(first_integer, max_new_tokens=400)[0].tolist()
    print(char_decoder(prediction, chars))
