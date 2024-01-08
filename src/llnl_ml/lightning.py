import pytorch_lightning as pl
import torch
import torchmetrics

from llnl_ml.model import get_model
from llnl_ml.optimizer import get_optimizer

from typing import Optional, Any, Union


class SegmentationLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for training a segmentation model
    """

    def __init__(
        self,
        model_name: Union[torch.nn.Module, str],
        input_channels: int = 1,
        output_channels: int = 1,
        lr: float = 1e-3,
        use_zero_grad: bool = False,
        schedular_type: str = "cosine_warmup",
        schedular_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(model_name, str):
            self.save_hyperparameters()
            self.model = get_model(
                model_name=model_name, input_channels=input_channels, output_channels=output_channels
            )
        else:
            self.save_hyperparameters(ignore=["model_name"])
            self.model = model_name
        self.lr = lr
        self.use_zero_grad = use_zero_grad
        self.scheduler_type = schedular_type
        self._scheduler_needs_epoch = False
        self._lr_scheduler_params = dict(mode="min", factor=0.1, patience=10, verbose=False)
        if isinstance(schedular_params, dict):
            self._lr_scheduler_params.update(schedular_params)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.acc_metric = torchmetrics.classification.BinaryAccuracy(threshold=0.5, multidim_average="global")
        self.jaccard_metric = torchmetrics.JaccardIndex(task="binary", threshold=0.5, ignore_index=0)
        self.dice_metric = torchmetrics.Dice(threshold=0.5)

    def forward(self, x, y: Optional[Any] = None):
        if y is not None:
            return self.model(x, y)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, targets = batch
        if self.model.calculates_loss:
            loss_dict = self.model(image, targets)
            self.log_dict(loss_dict, on_step=True, sync_dist=True, prog_bar=True)
            loss = sum(loss for loss in loss_dict.values())
        else:
            mask = targets["masks"].float()
            pred = self.model(image)
            # Squeeze out the single dimension channels dim to match mask
            # Shape [B, 1, H, W] -> [B, H, W]
            pred = pred.squeeze(dim=1)
            loss = self.loss_fn(pred, mask)
        self.log("train_loss", loss, on_step=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, targets = batch
        if isinstance(targets, (list, tuple)):
            # For each image's target, combine the individual components into a single mask
            mask = [target["masks"].sum(dim=0, keepdim=True) for target in targets]
            # Concatenate
            mask = torch.concatenate(mask, dim=0)
        else:
            mask = targets["masks"]
        pred = self.model(image)
        pred = pred.squeeze(dim=1)
        # Determine if we have logits or probabilities
        # If probabilities
        if pred.min() >= 0 and pred.max() <= 1.0:
            pred_prob = pred
            # Convert to pseudo logits [-1, 1] for calculating loss
            pred_logits = (pred - 0.5) * 2.0
        # If we have logits
        else:
            pred_prob = torch.nn.functional.sigmoid(pred)
            pred_logits = pred
        loss = self.loss_fn(pred_logits, mask.to(pred_logits))
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        # Metrics expect mask to be of type int, not float
        mask = mask.short()
        if (pred_prob.device != mask.device) or (pred_prob.device != self.acc_metric.device):
            print(f"Mismatch devices in Validation loop: ")
            print(f"\t{pred.device = }")
            print(f"\t{pred_prob.device = }")
            print(f"\t{pred_logits.device = }")
            print(f"\t{mask.device = }")
            print(f"\t{self.acc_metric.device = }")
        self.acc_metric(pred_prob, mask)
        self.jaccard_metric(pred_prob, mask)
        self.dice_metric(pred_prob, mask)
        self.log("val_dice", self.dice_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_jaccard", self.jaccard_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.acc_metric, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        image, targets = batch
        if isinstance(targets, (list, tuple)):
            # For each image's target, combine the individual components into a single mask
            mask = [target["masks"].sum(dim=0, keepdim=True) for target in targets]
            # Concatenate
            mask = torch.concatenate(mask, dim=0)
        else:
            mask = targets["masks"]
        pred = self.model(image)
        pred = pred.squeeze(dim=1)
        # Determine if we have logits or probabilities
        # If probabilities
        if pred.min() >= 0 and pred.max() <= 1.0:
            pred_prob = pred
            # Convert to pseudo logits [-1, 1] for calculating loss
            pred_logits = (pred - 0.5) * 2.0
        # If we have logits
        else:
            pred_prob = torch.nn.functional.sigmoid(pred)
            pred_logits = pred
        loss = self.loss_fn(pred_logits, mask.to(pred_logits))
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        # Metrics expect mask to be of type int, not float
        mask = mask.short()
        self.acc_metric(pred_prob, mask)
        self.jaccard_metric(pred_prob, mask)
        self.dice_metric(pred_prob, mask)
        self.log("test_dice", self.dice_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_jaccard", self.jaccard_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.acc_metric, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Any:
        optimizers, self._scheduler_needs_epoch = get_optimizer(
            parameters=self.parameters(),
            lr=self.lr,
            scheduler_type=self.scheduler_type,
            lr_scheduler_params=self._lr_scheduler_params,
            max_epochs=self.trainer.max_epochs,
            use_zero_grad=self.use_zero_grad,
        )

        return optimizers

    def lr_scheduler_step(self, scheduler: Any, metric: Optional[Any]) -> None:
        if metric is None:
            if self._scheduler_needs_epoch:
                scheduler.step(epoch=self.current_epoch)
            else:
                scheduler.step()  # type: ignore[call-arg]
        else:
            if self._scheduler_needs_epoch:
                scheduler.step(metric=metric, epoch=self.current_epoch)
            else:
                scheduler.step(metric)
