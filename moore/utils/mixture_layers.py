# Copyright 2021 The PODNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =====================================================================================
"""
Implementation is adapted from https://github.com/caisr-hh/podnn.

MOORE-HH version:
- OrthogonalLayer1D is now implemented with Householder-QR (via torch.linalg.qr).
- No Gram-Schmidt backend is kept in this codebase.
"""

import torch
import torch.nn as nn
from copy import deepcopy

n_models_global = 5
agg_out_dim = 3


class InputLayer(nn.Module):
    """
    InputLayer structures the data in a parallel form ready to be consumed by
    the upcoming parallel layers.
    """

    def __init__(self, n_models):
        """
        Args:
            n_models: number of individual models within the ensemble
        """
        super(InputLayer, self).__init__()
        self.n_models = n_models
        global n_models_global
        n_models_global = self.n_models

    def forward(self, x):
        """
        Args:
            x: input to the network as in standard deep neural networks.

        Returns:
            x_parallel: parallel form of x with shape [n_models, ...]
        """
        x_parallel = torch.unsqueeze(x, 0)
        x_parallel_next = torch.unsqueeze(x, 0)
        for _ in range(1, self.n_models):
            x_parallel = torch.cat((x_parallel, x_parallel_next), axis=0)

        return x_parallel


class ParallelLayer(nn.Module):
    """
    ParallelLayer creates a parallel layer from the structure of unit_model it receives.
    """

    def __init__(self, unit_model):
        """
        Args:
            unit_model: specifies the computational module each unit contains.
        """
        super(ParallelLayer, self).__init__()
        self.n_models = n_models_global
        self.model_layers = []
        for _ in range(self.n_models):
            for j in range(len(unit_model)):
                try:
                    unit_model[j].reset_parameters()
                except Exception:
                    pass
            self.model_layers.append(deepcopy(unit_model))
        self.model_layers = nn.ModuleList(self.model_layers)

    def forward(self, x):
        """
        Args:
            x: parallel input with shape [n_models, n_samples, dim]
               for fully connected outputs.

        Returns:
            parallel_output with shape [n_models, n_samples, dim]
        """
        parallel_output = self.model_layers[0](x[0])
        parallel_output = torch.unsqueeze(parallel_output, 0)
        for i in range(1, self.n_models):
            next_layer = self.model_layers[i](x[i])
            next_layer = torch.unsqueeze(next_layer, 0)
            parallel_output = torch.cat((parallel_output, next_layer), 0)

        return parallel_output


def orth_error_stats(basis):
    """
    Compute orthogonality error statistics for logging.

    Args:
        basis: [n_models, n_samples, dim]

    Returns:
        dict with Frobenius norm statistics of (V V^T - I), per sample.
    """
    x = basis.transpose(0, 1)  # [B, K, D]
    gram = torch.matmul(x, x.transpose(1, 2))  # [B, K, K]
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
    err = torch.linalg.norm(gram - eye, dim=(1, 2))  # [B]

    return {
        "orth/err_fro_mean": err.mean().item(),
        "orth/err_fro_max": err.max().item(),
        "orth/err_fro_p95": torch.quantile(err, 0.95).item(),
    }


class OrthogonalLayer1D(nn.Module):
    """
    OrthogonalLayer1D for MOORE-HH.

    This layer performs orthogonalization with batched reduced QR decomposition,
    whose standard implementation is based on Householder reflections.

    Input shape:
        [n_models, n_samples, dim]

    Output shape:
        [n_models, n_samples, dim]
    """

    def __init__(self, eps=1e-8, hh_canon_sign=True, hh_rank_tol=1e-6):
        super(OrthogonalLayer1D, self).__init__()
        self.eps = eps
        self.hh_canon_sign = hh_canon_sign
        self.hh_rank_tol = hh_rank_tol
        self.last_stats = {}

    def forward(self, x):
        """
        Args:
            x: [n_models, n_samples, dim]

        Returns:
            basis: [n_models, n_samples, dim]
        """
        # [K, B, D] -> [B, D, K]
        x1 = x.transpose(0, 1).transpose(1, 2)

        # Batched reduced QR
        q, r = torch.linalg.qr(x1, mode="reduced")

        diag_r = torch.diagonal(r, dim1=-2, dim2=-1)

        if self.hh_canon_sign:
            s = torch.sign(diag_r)
            s = torch.where(s == 0, torch.ones_like(s), s)
            q = q * s.unsqueeze(1)
            r = s.unsqueeze(-1) * r

        # [B, D, K] -> [K, B, D]
        basis = q.transpose(1, 2).transpose(0, 1)

        min_abs_diag = diag_r.abs().min()
        max_abs_diag = diag_r.abs().max()
        diag_ratio = max_abs_diag / (min_abs_diag + self.eps)
        rank_fail_rate = (diag_r.abs() < self.hh_rank_tol).float().mean()

        stats = orth_error_stats(basis)
        stats.update({
            "hh/min_abs_diagR": float(min_abs_diag.detach().cpu()),
            "hh/max_abs_diagR": float(max_abs_diag.detach().cpu()),
            "hh/diagR_ratio": float(diag_ratio.detach().cpu()),
            "hh/rank_fail_rate": float(rank_fail_rate.detach().cpu()),
            "hh/canon_sign": float(1.0 if self.hh_canon_sign else 0.0),
        })
        self.last_stats = stats

        return basis
