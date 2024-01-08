from typing import Any, Optional

import torch
from timm.scheduler import CosineLRScheduler
from torch.distributed.optim import ZeroRedundancyOptimizer


def get_optimizer(
    parameters: Any,
    lr: float = 1.0e-3,
    scheduler_type: str = "cosine_warmup",
    lr_scheduler_params: Optional[dict] = dict(),
    max_epochs: int = 10,
    use_zero_grad: bool = False,
):
    if use_zero_grad:
        optimizer = ZeroRedundancyOptimizer(parameters, optimizer_class=torch.optim.Adam, lr=lr)
    else:
        optimizer = torch.optim.Adam(parameters, lr=lr)

    scheduler_needs_epoch = False
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_params)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
        )
    elif scheduler_type == "cosine_warmup":
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=max_epochs,
            cycle_decay=0.5,
            lr_min=1e-14,
            warmup_t=min(max_epochs // 2, 5),
            warmup_lr_init=lr / 1000.0,
            cycle_limit=1,
        )
        scheduler_needs_epoch = True
    else:
        raise ValueError(f"Scheduler type {scheduler_type} not supported")
    scheduler_config = {
        "scheduler": scheduler,
        "interval": "epoch",
        "frequency": 1,
        "monitor": "train_loss",
    }
    return {"optimizer": optimizer, "lr_scheduler": scheduler_config}, scheduler_needs_epoch
