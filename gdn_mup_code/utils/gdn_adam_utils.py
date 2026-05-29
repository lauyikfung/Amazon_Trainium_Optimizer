"""
Fine-grained parameter groups for GDN + AdamW muP training.

For AdamW, hidden weights use the muP hidden LR multiplier, embeddings use the
embedding multiplier, and GDN scalar parameters (A_log, dt_bias) keep the base LR
with no weight decay.
"""

from __future__ import annotations

import torch.nn as nn


def get_gdn_adam_param_groups(
    model: nn.Module,
    weight_decay: float,
    *,
    hidden_lr_mult: float = 1.0,
    embedding_lr_mult: float = 1.0,
) -> list[dict]:
    """Build AdamW param groups for a Trainium GDN model following the muP spec."""
    embedding_params: set[nn.Parameter] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters():
                if isinstance(p, nn.Parameter):
                    embedding_params.add(p)

    scalar_params: set[nn.Parameter] = set()
    for module in model.modules():
        if hasattr(module, "A_log") and hasattr(module, "dt_bias"):
            if isinstance(module.A_log, nn.Parameter):
                scalar_params.add(module.A_log)
            if isinstance(module.dt_bias, nn.Parameter):
                scalar_params.add(module.dt_bias)

    no_wd_params: set[nn.Parameter] = set()
    for module in model.modules():
        if module.__class__.__name__ in {"RMSNorm", "RMSNormLinear", "FusedRMSNormGated", "TorchRMSNormGated"}:
            for p in module.parameters():
                if isinstance(p, nn.Parameter):
                    no_wd_params.add(p)
    for _, param in model.named_parameters():
        if isinstance(param, nn.Parameter) and getattr(param, "_no_weight_decay", False):
            no_wd_params.add(param)

    hidden_decay: list[nn.Parameter] = []
    hidden_no_decay: list[nn.Parameter] = []
    emb_list: list[nn.Parameter] = []
    scalar_list: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in embedding_params:
            emb_list.append(param)
        elif param in scalar_params:
            scalar_list.append(param)
        elif param in no_wd_params or name.endswith("bias"):
            hidden_no_decay.append(param)
        else:
            hidden_decay.append(param)

    groups: list[dict] = []
    if hidden_decay:
        groups.append(
            {"params": hidden_decay, "weight_decay": weight_decay, "lr_mult": hidden_lr_mult, "name": "hidden_decay"}
        )
    if hidden_no_decay:
        groups.append(
            {"params": hidden_no_decay, "weight_decay": 0.0, "lr_mult": hidden_lr_mult, "name": "hidden_no_decay"}
        )
    if emb_list:
        groups.append({"params": emb_list, "weight_decay": 0.0, "lr_mult": embedding_lr_mult, "name": "embedding"})
    if scalar_list:
        groups.append({"params": scalar_list, "weight_decay": 0.0, "lr_mult": 1.0, "name": "gdn_scalar"})

    return groups
