""" Stub module to set device
"""
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
