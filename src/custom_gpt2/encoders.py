"""Encoders
"""
from typing import List


def char_encoder(string: str, code_book: List[str]) -> List[int]:
    """Character-level tokeniser, with small code book.

    Take a string and return a list of integers.
    TODO Report char not present in code book

    :return:
    """
    str_to_int = {s: i for i, s in enumerate(code_book)}

    try:
        encoded = [str_to_int[s] for s in string]
    except KeyError:
        raise KeyError("Char in string not present in code book")

    return encoded


def char_decoder(integers: List[int], code_book: List[str]) -> str:
    """Character-level tokeniser, with small code book.

    Take a list of integers and return a string.
    TODO Report char not present in code book

    :return:
    """
    int_to_str = {i: s for i, s in enumerate(code_book)}

    try:
        decoded = "".join(int_to_str[i] for i in integers)
    except KeyError:
        raise KeyError("Char in string not present in code book")

    return decoded
