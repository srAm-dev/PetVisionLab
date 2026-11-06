import math
import torch

def scaled_dot_product_attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights
