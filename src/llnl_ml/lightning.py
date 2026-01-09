import pytorch_lightning as pl
import torch
import torchmetrics

from torchmetrics.classification import BinaryMatthewsCorrCoef
from torchmetrics.segmentation import DiceScore, HausdorffDistance
from torchmetrics.functional.segmentation import dice_score
from torchvision.transforms.functional import to_pil_image

from llnl_ml.model import get_model
from llnl_ml.optimizer import get_optimizer
from timm.scheduler.scheduler import Scheduler as TIMMScheduler

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
        max_hausdorff_size: int = -1,
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
        self.acc_metric = torchmetrics.Accuracy(task="binary", threshold=0.5, multidim_average="global")
        self.jaccard_metric = torchmetrics.JaccardIndex(task="binary", threshold=0.5, ignore_index=0)
        self.dice_metric = DiceScore(2, include_background=False, input_format="one-hot", average="macro")
        self.mathew_coer_metric = BinaryMatthewsCorrCoef(ignore_index=0, threshold=0.5)
        self.hausdorff_metric = HausdorffDistance(2, include_background=False, input_format="one-hot")
        self._max_hausdorff_size = max_hausdorff_size

        # Aggregates the first 5 test images to be logged
        self.test_log_images = list()
        self.test_log_images_raw = list()

        # Export test scores for individual files
        self.test_image_metrics = list()

    def forward(self, x, y: Optional[Any] = None):
        if y is not None:
            return self.model(x, y)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, targets = batch
        
        # All models now return loss dict
        pred_logits, loss_dict = self.model(image, targets)
        
        # Compute total loss
        total_loss = sum(loss for loss in loss_dict.values())
        
        # Add total_loss to dict for logging
        loss_dict_with_total = {**loss_dict, "total_loss": total_loss}
        
        # Log all losses
        self.log_dict(loss_dict_with_total, on_step=True, sync_dist=True, prog_bar=True)
        
        return total_loss

    # Helper functions used in val and test
    def _get_eval_pred_and_mask(self, batch, batch_idx):
        # Get prediction and losses from model
        images, targets = batch
        pred_logits, losses = self.model(images, targets)

        # Extract the binary mask for computing metrics against the prediction
        if isinstance(targets, (list, tuple)):
            # For each image's target, combine the individual components into a single mask
            mask = [target["masks"].sum(dim=0, keepdim=True) for target in targets]
            # Concatenate
            mask = torch.concatenate(mask, dim=0)
        else:
            mask = targets["masks"]

        # Metrics expect mask to be of type int, not float
        mask = mask.long()
        # Send logits through sigmoid to create probabilities
        pred_prob = torch.nn.functional.sigmoid(pred_logits)
        
        return images, mask, pred_prob, pred_logits, losses


    def validation_step(self, batch, batch_idx):
        # Get prediction and losses from model
        _, mask, pred_prob, _, losses = self._get_eval_pred_and_mask(batch, batch_idx)
        pred_mask = (pred_prob > 0.5).long()

        # Add total loss and prepend val_ to all loss keys
        losses["total_loss"] = sum(losses.values())
        losses = {f"val_{key}": value for key, value in losses.items()}

        self.log_dict(losses, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        self.acc_metric(pred_prob, mask)
        self.jaccard_metric(pred_prob, mask)
        self.mathew_coer_metric(pred_prob, mask)
        self.dice_metric(pred_mask, mask)
        if self._max_hausdorff_size > -1:
            pred_mask = torch.nn.functional.interpolate(pred_mask.to(torch.uint8), size=(self._max_hausdorff_size, self._max_hausdorff_size), mode="nearest-exact")
            mask = torch.nn.functional.interpolate(mask.to(torch.uint8), size=(self._max_hausdorff_size, self._max_hausdorff_size), mode="nearest-exact")
        self.hausdorff_metric(pred_mask.cpu(), mask.cpu())

        self.log("val_dice", self.dice_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_jaccard", self.jaccard_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.acc_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mathews", self.mathew_coer_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_hausdorff", self.hausdorff_metric, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Get prediction and losses from model
        images, mask, pred_prob, pred_logits, losses = self._get_eval_pred_and_mask(batch, batch_idx)
        pred_mask = (pred_prob > 0.5).long()

        # Add total loss and prepend test_ to all loss keys
        losses["total_loss"] = sum(losses.values())
        losses = {f"test_{key}": value for key, value in losses.items()}
        
        self.log_dict(losses, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        
        self.acc_metric(pred_prob, mask)
        self.jaccard_metric(pred_prob, mask)
        self.mathew_coer_metric(pred_prob, mask)
        self.dice_metric(pred_mask, mask)
        resized_mask = mask
        if self._max_hausdorff_size > -1:
            pred_mask = torch.nn.functional.interpolate(pred_mask.to(torch.uint8), size=(self._max_hausdorff_size, self._max_hausdorff_size), mode="nearest-exact")
            resized_mask = torch.nn.functional.interpolate(mask.to(torch.uint8), size=(self._max_hausdorff_size, self._max_hausdorff_size), mode="nearest-exact")
        self.hausdorff_metric(pred_mask.cpu(), resized_mask.cpu())

        self.log("test_dice", self.dice_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_jaccard", self.jaccard_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.acc_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_mathews", self.mathew_coer_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_hausdorff", self.hausdorff_metric, on_step=False, on_epoch=True, prog_bar=True)

        # Store scores for logging later
        # To ensure a one to one mapping of metrics to images we will re-compute using the functional interface
        if isinstance(batch[1], list):
            image_names = [target["image_name"] for target in batch[1]]
        else:
            image_names = batch[1]["image_name"]

        for img_name, img, img_pred, gt_mask in zip(image_names, images, pred_prob, mask):
            self.test_image_metrics.append(
                {
                    "image_name": img_name,
                    "dice": dice_score((img_pred > 0.5).long(), gt_mask, num_classes=2, include_background=False, average="macro").cpu().item(),
                    "jaccard": torchmetrics.functional.jaccard_index((img_pred > 0.5).long(), gt_mask, task="binary")
                    .cpu()
                    .item(),
                    "acc": torchmetrics.functional.accuracy((img_pred > 0.5).long(), gt_mask, task="binary").cpu().item(),
                }
            )
        # If this is one of the first 5 test images, save out the mask and image
        if self.trainer.is_global_zero and len(self.test_log_images) < 5:
            # Get up to the first 5 images and masks
            for img, img_pred, gt_mask in zip(images, pred_prob, mask):
                np_pred_mask = (img_pred > 0.5).short().cpu().permute((1, 2, 0)).numpy()
                np_gt_mask = gt_mask.cpu().permute((1, 2, 0)).numpy()
                pil_image = to_pil_image(img.cpu())
                # Store raw image data for logger-agnostic artifact logging
                self.test_log_images.append({
                    "image": pil_image,
                    "pred_mask": np_pred_mask,
                    "gt_mask": np_gt_mask,
                })
                self.test_log_images_raw.append(
                    (img.cpu().squeeze().numpy(), np_gt_mask, img_pred.cpu().squeeze().numpy())
                )

                if len(self.test_log_images) >= 5:
                    break

    def configure_optimizers(self) -> Any:
        optimizers, self._scheduler_needs_epoch = get_optimizer(
            parameters=self.parameters(),
            lr=self.lr,
            scheduler_type=self.scheduler_type,
            lr_scheduler_params=self._lr_scheduler_params,
            max_iters=int(self.trainer.estimated_stepping_batches),
        )

        return optimizers

    def lr_scheduler_step(self, scheduler: Any, metric: Optional[Any]) -> None:
        if isinstance(scheduler, TIMMScheduler):
            # TIMM schedulers expect the step number (0-indexed)
            # For step-based scheduling, use global_step
            scheduler.step(epoch=self.trainer.global_step)
        else:
            super().lr_scheduler_step(scheduler=scheduler, metric=metric)
