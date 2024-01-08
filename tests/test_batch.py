from unittest.mock import patch

import torch

from custom_gpt2.batch import get_batch


def test_get_batch():
    block_size = 3
    batch_size = 3
    data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

    with patch("custom_gpt2.batch.torch.randint") as mock_randint:
        # Max starting index one can specify
        upper_bound = len(data) - block_size

        # Mock return value of torch.randint
        mock_randint.return_value = torch.tensor([4, 1, 3])
        assert max(mock_randint.return_value) <= upper_bound, (
            "Mock index i will cause (i + block_size) to be out of " "bounds"
        )
        x_ref = torch.tensor([[5, 6, 7], [2, 3, 4], [4, 5, 6]])
        y_ref = torch.tensor([[6, 7, 8], [3, 4, 5], [5, 6, 7]])

        x, y = get_batch(data, block_size, batch_size)

    ix = mock_randint.return_value
    for i in range(batch_size):
        assert (
            x[i, 0] == data[ix[i]]
        ), "First element of a row should correspond to data at index ix, per batch"

    # y data should be offset by 1 w.r.t x data
    # Not the most intuitive assertion
    for i in range(batch_size):
        assert torch.equal(y[i, :-1], x[i, 1:])

    assert x.shape == torch.Size([batch_size, block_size])
    assert y.shape == torch.Size([batch_size, block_size])
    assert torch.equal(x, x_ref)
    assert torch.equal(y, y_ref)
