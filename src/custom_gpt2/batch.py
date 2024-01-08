"""Batching
"""
from typing import Tuple

import torch


def get_batch(
    data, block_size: int, batch_size: int
) -> Tuple[torch.stack, torch.stack]:
    """Batch data into random blocks of inputs and outputs
    Note, can easily switch the block_size and batch_size args here, when calling
    - Consider stronger typing

    :return:
    """
    n_random = batch_size
    # Total amount of data minus the block size
    upper_bound = len(data) - block_size
    # This gives `n_random` blocks of size `block_size`
    ix = torch.randint(0, upper_bound, (n_random,))
    # Stack put each block into a row
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    assert list(x.shape) == [batch_size, block_size]
    return x, y
