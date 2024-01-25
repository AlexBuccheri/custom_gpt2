"""Batching
"""
from typing import Tuple

import torch

from custom_gpt2.cuda_settings import device


def get_batch(
    data, block_size: int, batch_size: int
) -> Tuple[torch.stack, torch.stack]:
    """Batch data into random blocks of inputs and outputs

    Training for LLMs needs block_size + 1, as we're training to make predictions on the block_size,
    but a prediction requires the ith + 1 character

    Note, can easily switch the block_size and batch_size args here, when calling
    - Consider stronger typing

    :return:
    """
    # Total amount of data minus the block size
    upper_bound = len(data) - block_size
    # This gives the number of random blocks of size `block_size` i.e. the batch_size
    ix = torch.randint(0, upper_bound, (batch_size,))
    # Stack puts each block into a row
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    assert list(x.shape) == [batch_size, block_size]
    x.to(device), y.to(device)
    return x, y
