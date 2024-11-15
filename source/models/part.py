"""Particle transformer."""

from typing import Optional

import torch
import torch.nn as nn


@torch.jit.script
def cms_to_cartesian(pt: torch.Tensor, eta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Convert the CMS coordinates to Cartesian coordinates.

    Args:
        pt : torch.Tensor
            Transverse momentum.
        eta : torch.Tensor
            Pseudorapidity.
        phi : torch.Tensor
            Azimuthal angle.
    Returns:
        torch.Tensor
            Cartesian coordinates.
    """

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)

    return torch.stack((px, py, pz), dim=-1)


@torch.jit.script
def prepare_interaction(x: torch.Tensor, include_mass: bool, eps: float = 1E-9) -> torch.Tensor:
    """Prepare the features for interaction matrix U.

    Args:
        x : torch.Tensor
            Input tensor of shape (N, L, D), where N is the batch size,
            L is the number of particles, and D is the feature dimension
            corresponding to (pt, deta, dphi).
        include_mass : bool
            Whether to include the mass of the particles.
    Returns:
        torch.Tensor
            Output tensor of shape (N, 4, L, L), where N is the batch
            size, L is the number of particles, and 4 is the size of the
            interaction matrix U.
    """

    # Unsqueeze for broadcasting.
    x_i = x.unsqueeze(-2)  # (N, L, D) -> (N, L, 1, D)
    x_j = x.unsqueeze(-3)  # (N, L, D) -> (N, 1, L, D)

    # We only use pt, deta, dphi.
    log_pt_i, deta_i, dphi_i = x_i[..., :3].unbind(dim=-1)  # (N, L, 1) * 3
    log_pt_j, deta_j, dphi_j = x_j[..., :3].unbind(dim=-1)  # (N, 1, L) * 3

    # The transverse momentum is stored in log scale.
    pt_i = torch.exp(log_pt_i)  # (N, L, 1)
    pt_j = torch.exp(log_pt_j)  # (N, 1, L)

    # Calculate the interaction matrix
    delta = torch.sqrt((deta_i - deta_j) ** 2 + ((dphi_i - dphi_j + torch.pi) % (2 * torch.pi) - torch.pi) ** 2).clamp(min=eps)  # (N, L, L)
    kt = (torch.minimum(pt_i, pt_j) * delta).clamp(min=eps)  # (N, L, L)
    z = (torch.minimum(pt_i, pt_j) / (pt_i + pt_j)).clamp(min=eps)  # (N, L, L)

    # Calculate the mass of the particles.
    if include_mass:
        E_i = x_i[..., 3]  # (N, L, 1)
        E_j = x_j[..., 3]  # (N, 1, L)

        px_i, py_i, pz_i = cms_to_cartesian(pt_i, deta_i, dphi_i).unbind(dim=-1)  # (N, L, 1) * 3
        px_j, py_j, pz_j = cms_to_cartesian(pt_j, deta_j, dphi_j).unbind(dim=-1)  # (N, 1, L) * 3

        m_2 = ((E_i + E_j) ** 2 - (px_i + px_j) ** 2 - (py_i + py_j) ** 2 - (pz_i + pz_j) ** 2).clamp(min=eps)  # (N, L, L)

        return torch.log(torch.stack((delta, kt, z, m_2), dim=-3))  # (N, 4, L, L)
    else:
        return torch.log(torch.stack((delta, kt, z), dim=-3))  # (N, 3, L, L)


class ParticleFeatureEmbedding(nn.Module):

    def __init__(self, input_dim: int, embedding_dims: list) -> None:

        super().__init__()

        dims = [input_dim] + embedding_dims
        dims = [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

        layers = []
        for _input_dim, _embed_dim in dims:
            layers.append(nn.LayerNorm(_input_dim))
            layers.append(nn.Linear(_input_dim, _embed_dim))
            layers.append(nn.GELU())

        self.embedding = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.embedding(x)  # (N, L, D) -> (N, L, E)


class InteractionMatrixEmbedding(nn.Module):

    def __init__(self, input_dim: int, embedding_dims: list) -> None:

        super().__init__()

        dims = [input_dim] + embedding_dims
        dims = [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

        layers = []
        for _input_dim, _embed_dim in dims:
            layers.append(nn.Conv2d(_input_dim, _embed_dim, kernel_size=1))
            layers.append(nn.GELU())

        self.embedding = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.embedding(x)  # (N, C, L, L) -> (N, H, L, L)

        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: int = 0.1, bias: bool = False, isheadscale: bool = True) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for queries, keys, and values
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Softmax for calculating QK^T
        self.softmax = nn.Softmax(dim=-1)

        # Linear projection for the output of attention heads
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Head scale from https://arxiv.org/pdf/2110.09456.pdf
        self.gamma = nn.Parameter(torch.ones(num_heads), requires_grad=True) if isheadscale else None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Linear projections.
        q = self.q_linear(query)  # (N, L, E) | (N, 1, E)
        k = self.k_linear(key)   # (N, L, E) | (N, 1 + L, E)
        v = self.v_linear(value)  # (N, L, E) | (N, 1 + L, E)

        # Split into multiple heads. (E = H x D)
        q = q.view(*q.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)  # (N, L, E) -> (N, L, H, D) -> (N, H, L, D) | (N, H, 1, D)
        k = k.view(*k.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)  # (N, L, E) -> (N, L, H, D) -> (N, H, L, D) | (N, H, 1 + L, D)
        v = v.view(*v.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)  # (N, L, E) -> (N, L, H, D) -> (N, H, L, D) | (N, H, 1 + L, D)

        # Scaled Dot-Product Attention.
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (N, H, L, D) x (N, H, D, L) = (N, H, L, L) | (N, H, 1, 1 + L)

        # Apply mask on scores.
        if key_padding_mask is not None:
            scores_mask = key_padding_mask.unsqueeze(-2).unsqueeze(-2)  # (N, 1, 1, L) | (N, 1, 1, 1 + L)
            scores = scores.masked_fill(scores_mask, float('-inf'))  # (N, H, L, L) | (N, H, 1, 1 + L)

            v_mask = key_padding_mask.unsqueeze(-2).unsqueeze(-1)  # (N, 1, L, 1) | (N, 1, 1 + L, 1)
            v = v.masked_fill(v_mask, 0.0)  # (N, H, L, D) | (N, H, 1 + L, D)

        # Apply interaction U.
        if attn_mask is not None:
            scores = scores + attn_mask

        # Softmax (scores -> attention weights).
        attn_weights = self.softmax(scores)  # (N, H, L, L)
        attn_weights = self.dropout(attn_weights)  # (N, H, L, L)

        # Weighted sum of values.
        weighted_sum = torch.matmul(attn_weights, v)  # (N, H, L, L) x (N, H, L, D) = (N, H, L, D)

        if self.gamma is not None:
            weighted_sum = torch.einsum('bhtd,h->bhtd', weighted_sum, self.gamma)

        # Concatenate attention heads and apply the final linear projection.
        weighted_sum = weighted_sum.permute(0, 2, 1, 3).contiguous().view(q.size(0), -1, self.embed_dim)  # (N, H, L, D) -> (N, L, H, D) -> (N, L, E)
        output = self.out_linear(weighted_sum)  # (N, L, E)

        return output


class AttentionBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, fc_dim: int, dropout: float = 0.1, **kwargs) -> None:
        super().__init__()

        self.attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.pre_attn_layerNorm = nn.LayerNorm(embed_dim)
        self.post_attn_layerNorm = nn.LayerNorm(embed_dim)

        self.pre_fc_layerNorm = nn.LayerNorm(embed_dim)
        self.post_fc_layerNorm = nn.LayerNorm(fc_dim)

        self.fc1 = nn.Linear(embed_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, embed_dim)
        self.act = nn.GELU()

        self.act_dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.lambda_residual = nn.Parameter(torch.ones(embed_dim), requires_grad=True)

    def forward(self, x: torch.Tensor, x_clt: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if x_clt is not None:  # Class Attention Block.

            # Add mask for class token which is added in the key.
            with torch.no_grad():
                if key_padding_mask is not None:
                    clt_padding_mask = torch.zeros_like(key_padding_mask[..., 0]).unsqueeze(-1)  # (N, 1)
                    key_padding_mask = torch.cat((clt_padding_mask, key_padding_mask), dim=-1)  # (N, 1) + (N, L) = (N, 1 + L)

            # First residual values.
            residual = x_clt

            # Multi-head attention (MHA).
            x = torch.cat((x_clt, x), dim=-2)  # (N, 1, E) + (N, L, E) = (N, 1 + L, E)
            x = self.pre_attn_layerNorm(x)
            x = self.attn(query=x_clt, key=x, value=x, key_padding_mask=key_padding_mask)  # (N, 1, E)

        else:  # Particle Attention Block.

            # First residual values.
            residual = x

            # Particle Multi-head attention (P-MHA).
            x = self.pre_attn_layerNorm(x)
            x = self.attn(query=x, key=x, value=x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        x = self.post_attn_layerNorm(x)
        x = self.dropout1(x)

        # First residual connection.
        x = x + residual

        # Second residual values.
        residual = x

        # Fully connected layers after first residual connection.
        x = self.pre_fc_layerNorm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.act_dropout(x)
        x = self.post_fc_layerNorm(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        # Resudual scaling.
        residual = torch.mul(self.lambda_residual, residual)

        # Second residual connection.
        x = x + residual

        return x


class ParticleTransformer(nn.Module):
    def __init__(self, score_dim: int, parameters: dict):
        """Particle Transformer.

        Args:
            score_dim : int
                Dimension of final output.
            parameters : dict
                Hyperparameters for the model, see `configs/benchmark.yaml`.
        """
        super().__init__()

        # Particle Embedding.
        self.par_embedding = ParticleFeatureEmbedding(
            input_dim=parameters['ParEmbed']['input_dim'],
            embedding_dims=parameters['ParEmbed']['embed_dim'],
        )

        # Interaction Embedding.
        self.include_mass = parameters['IntEmbed']['include_mass']
        self.int_embedding = InteractionMatrixEmbedding(
            input_dim=parameters['IntEmbed']['input_dim'],
            embedding_dims=parameters['IntEmbed']['embed_dim'] + [parameters['ParAtteBlock']['num_heads']],
        )

        # Embedding dimension used for all attention blocks and fully connected layers.
        attn_embed_dim = parameters['ParEmbed']['embed_dim'][-1]

        # Particle Attention Blocks.
        self.par_attn_blocks = nn.ModuleList([
            AttentionBlock(
                embed_dim=attn_embed_dim,
                **parameters['ParAtteBlock']
            ) for _ in range(parameters['num_ParAtteBlock'])
        ])

        # Class Attention Blocks.
        self.class_attn_blocks = nn.ModuleList([
            AttentionBlock(
                embed_dim=attn_embed_dim,
                **parameters['ClassAtteBlock']
            ) for _ in range(parameters['num_ClassAtteBlock'])
        ])

        # Class token (used in Class Attention Blocks).
        self.class_token = nn.Parameter(torch.zeros(1, 1, attn_embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.class_token)

        self.layerNorm = nn.LayerNorm(attn_embed_dim)

        self.fc = nn.Sequential(nn.Linear(attn_embed_dim, attn_embed_dim), nn.ReLU(), nn.Dropout(0.1))

        self.final_layer = nn.Linear(attn_embed_dim, score_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x : torch.Tensor
                    Input tensor of shape (N, L, D), where N is the batch size, 
                    L is the sequence length, and D is the feature dimension.
        """

        with torch.no_grad():
            # Mask for padding particles.
            key_padding_mask = torch.isnan(x[..., 0])  # (N, L)

            # Particle features.
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)  # (N, L, D)

            # Calculate interaction matrix.
            U_mask = (key_padding_mask.unsqueeze(-1) | key_padding_mask.unsqueeze(-2))  # (N, L, L)
            U_mask = U_mask.unsqueeze(-3)  # (N, 1, L, L)
            U = prepare_interaction(x, self.include_mass)  # (N, 4, L, L)
            U = U.masked_fill(U_mask, 0.0)  # (N, 4, L, L)

        # Particle and interaction embedding.
        x = self.par_embedding(x)  # (N, L, E)
        U = self.int_embedding(U)  # (N, H, L, L)

        # Particle Attention Block
        for block in self.par_attn_blocks:
            x = block(x, x_clt=None, attn_mask=U, key_padding_mask=key_padding_mask)  # (N, L, E)

        # Class Attention Block
        class_token = self.class_token.expand(x.size(0), -1, -1)  # (N, 1, E)
        for block in self.class_attn_blocks:
            class_token = block(x, x_clt=class_token, key_padding_mask=key_padding_mask)  # (N, L, E)

        # Layer normalization
        class_token = self.layerNorm(class_token).squeeze(1)

        # Fully connected layer
        class_token = self.fc(class_token)
        class_token = self.final_layer(class_token)

        return class_token
