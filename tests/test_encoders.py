from typing import List

import pytest

from custom_gpt2.encoders import char_decoder, char_encoder


@pytest.fixture()
def basic_code_book() -> List[str]:
    chars_str = (
        "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )
    return [s for s in chars_str]


def test_char_encode_decode(basic_code_book):
    # Encode, then decode to return original string

    string = "What's Up"
    assert (
        char_decoder(char_encoder(string, basic_code_book), basic_code_book)
        == string
    )

    string = "j3fo.3-\n"
    assert (
        char_decoder(char_encoder(string, basic_code_book), basic_code_book)
        == string
    )

    with pytest.raises(
        KeyError, match="Char in string not present in code book"
    ):
        string = "Some invalid chars: 4+="
        char_decoder(char_encoder(string, basic_code_book), basic_code_book)
