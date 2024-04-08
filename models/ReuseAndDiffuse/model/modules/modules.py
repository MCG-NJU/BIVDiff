import math

import torch


def get_sin_pos_embedding(embed_dim, seq_len):
    """
    :param embed_dim: dimension of the model
    :param seq_len: length of positions
    :return: [length, embed_dim] position matrix
    """
    if embed_dim % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(embed_dim)
        )
    pe = torch.zeros(seq_len, embed_dim)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float)
        * -(math.log(10000.0) / embed_dim)
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe
