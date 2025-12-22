from typing import Any, Optional

import torch
from timm.scheduler.scheduler_factory import create_scheduler_v2

def get_optimizer(
    parameters: Any,
    lr: float = 1.0e-3,
    scheduler_type: str = "cosine",
    lr_scheduler_params: Optional[dict] = None,
    max_iters: int = 1000,
):
    if lr_scheduler_params is None:
        lr_scheduler_params = {}
        
    optimizer = torch.optim.AdamW(parameters, lr=lr)

    # Map scheduler types to ensure compatibility
    scheduler_map = {
        "cosine_warmup": "cosine",
    }
    if scheduler_type in scheduler_map:
        scheduler_type = scheduler_map.get(scheduler_type, scheduler_type)

    try:
        scheduler, _ = create_scheduler_v2(
            optimizer=optimizer,
            sched=scheduler_type,
            num_epochs=max_iters,  # "epochs" are actually iterations
            step_on_epochs=True,  # Force epoch-based (but we'll step with iteration numbers)
            updates_per_epoch=1,  # Not used when step_on_epochs=True
            warmup_epochs=min(max_iters // 2, 400),  # Warmup "epochs" are actually iterations
            warmup_lr=lr / 1000.0,
        )
    except Exception as e:
        print(f"Warning: Failed to create scheduler '{scheduler_type}': {e}")
        print("Falling back to no scheduler")
        scheduler = None

    if scheduler is not None:
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "train_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}, True
    else:
        # Return optimizer only if scheduler creation failed
        return {"optimizer": optimizer}, False
