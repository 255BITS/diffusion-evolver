import argparse
import os
import shutil
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file

def merge_tensors(tensor1, tensor2, p):
    # Calculate the delta of the weights
    delta = tensor2 - tensor1
    # Generate the mask m^t from Bernoulli distribution
    m = torch.bernoulli(torch.full(delta.shape, p)).to(delta.dtype)
    # Apply the mask to the delta to get δ̃^t
    delta_tilde = m * delta
    # Scale the masked delta by the dropout rate to get δ̂^t
    delta_hat = delta_tilde / (1 - p)
    return delta_hat

def merge_safetensors(file_path1, file_path2, p, lambda_val):
    merged_tensors = {}

    with safe_open(file_path1, framework="pt", device="cpu") as f1, safe_open(file_path2, framework="pt", device="cpu") as f2:
        keys1 = set(f1.keys())
        keys2 = set(f2.keys())
        common_keys = keys1.intersection(keys2)

        for key in common_keys:
            tensor1 = f1.get_tensor(key)
            tensor2 = f2.get_tensor(key)
            merged_tensors[key] = tensor1 + lambda_val * merge_tensors(tensor1, tensor2, p)

    return merged_tensors
