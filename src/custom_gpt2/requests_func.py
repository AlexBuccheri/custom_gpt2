""" Wrappers around requests
"""
import enum

import requests


class RequestCode(enum.Enum):
    SUCCESS = 200
    CREATED = 2
    BAD_REQUEST = 400
    NOT_FOUND = 401
    SERVER_ERROR = 500


def request_data(url) -> str:
    """Request data from a URL
    :return:
    """
    response = requests.get(url)

    if response.status_code == RequestCode.SUCCESS.value:
        return response.content.decode("utf-8")
    else:
        raise requests.RequestException(
            f"Failed to download file. Status code: {response.status_code}"
        )
