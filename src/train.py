import argparse
import logging
import os

import boto3
import pytorch_lightning as pl
import torch
import yaml
from pathlib import Path
from matplotlib import pyplot as plt

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from typing import Optional, Tuple

from llnl_ml.model import count_parameters
from llnl_ml.data import get_data_loaders
from llnl_ml.lightning import SegmentationLightningModule
from llnl_ml.util import str2bool, str2intlist

from pytorch_lightning.loggers import WandbLogger


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for U-Net segmentation model. "
        "Pass model specific parameters as command line arguments with 'MODEL' prefix. "
        "For example '--MODEL.in_channels 1' or 'MODEL.pretrained 0`"
    )

    parser.add_argument("--model_name", type=str, default="UNet", help="Name of the model to use for training")

    parser.add_argument(
        "--image_folder",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN_IMAGE"],
        help="Path to the folder containing images",
    )
    parser.add_argument(
        "--mask_folder",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN_MASK"],
        help="Path to the folder containing masks",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help=(
            "Path or S3 uri to json file containing the images and masks filenames for the train, val, "
            "and test splits. Will prepend the image and mask folder names to filenames in the split. "
            "If not provided, will create splits randomly from all available images in the image and mask folders."
        ),
    )
    parser.add_argument(
        "--train_count",
        type=int,
        default=-1,
        help="Maximum number of training images to use. Use -1 to use all images.",
    )
    parser.add_argument(
        "--output_data_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "output/data"),
        help="Location for saving output artifacts like logs, metrics, etc.",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DIR", "output/"),
        help="Location for saving tensorboard logs",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "output/model/"),
        help="Location for saving the model",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/opt/ml/checkpoints",
        help=(
            "Location to save intermediate training checkpoints. "
            "If directory is not empty, will restart training using the last.ckpt. "
            "Used by spot instance training to allow for graceful start/resume cycles."
        ),
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help=(
            "S3Uri or path to .ckpt file containing weights to load into model. "
            "NOTE: Checkpoints in the checkpoints directory will override these weights when they "
            "load for training resumption."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument("--lr_schedular", type=str, default="cosine_warmup", help="LR Schedular")
    parser.add_argument(
        "--image_mode",
        type=str,
        choices=["RGB", "grayscale"],
        default="grayscale",
        help="Image mode for reading images (RGB or grayscale)",
    )
    parser.add_argument("--use_zero", type=str2bool, default=False, help="Enable ZeRO optimizer")
    parser.add_argument("--use_amp", type=str2bool, default=False, help="Enable Mixed Precision")
    parser.add_argument("--use_random_resize", type=str2bool, default=False, help="Use random resized crops")
    parser.add_argument(
        "--use_random_crop",
        type=str2bool,
        default=False,
        help="Uses a random crop without resizing. Mutually exclusive with use_random_resize. "
        "If both set, will only do random resize",
    )
    parser.add_argument("--image_size", type=int, default=512, help="Image size for training")
    parser.add_argument(
        "--use_random_rotation", type=str2bool, default=False, help="Use random rotation augment during training"
    )
    parser.add_argument(
        "--color_jitter", type=float, default=0.0, help="Value of color jitter to apply during training"
    )
    parser.add_argument("--val_image_size", type=int, default=None, help="Image size for validation set")
    parser.add_argument("--val_batch_size", type=int, default=None, help="Batch size for validation loader")
    parser.add_argument("--test_image_size", type=int, default=None, help="Image size for test set")
    parser.add_argument("--test_batch_size", type=int, default=None, help="Batch size for test loader")
    parser.add_argument(
        "--center_crop",
        type=str2bool,
        default=False,
        help="If True, performs offset center crop of image and mask prior to transforms",
    )
    parser.add_argument("--center_crop_size", type=int, default=1200, help="Size of the center crop")
    parser.add_argument(
        "--center_crop_offset",
        type=str2intlist,
        default=[-60, -50],
        help="Center off set as list [Y, X] or [Height, Width] order",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="project_name2",
        help="  ",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="run_name2",
        help="  ",
    )

    args, model_params = parser.parse_known_args()
    return args, model_params


def download_weights(s3_uri: str) -> str:
    """
    Checks if s3_uri is a s3 uri or a local path. If s3 uri, will download the file locally to a temp directory
    and return the temp local path.

    :param s3_uri: String of either local path or s3 uri (s3://bucket/path/to/file.ckpt)
    :return: String of local weights location
    """
    if not s3_uri.startswith("s3://"):
        return s3_uri

    bucket, key = s3_uri.removeprefix("s3://").split("/", 1)

    s3 = boto3.client("s3")
    local_file = f"/tmp/{Path(s3_uri).name}"
    s3.download_file(bucket, key, local_file)

    return local_file


def main(
    model_name: str,
    image_folder: str,
    mask_folder: str,
    output_data_dir: str,
    tensorboard_dir: str,
    model_dir: str,
    checkpoint_dir: str = "/opt/ml/checkpoints",
    pretrained_weights: Optional[str] = None,
    split_file: Optional[str] = None,
    train_count: int = -1,
    model_params: Optional[dict] = None,
    num_workers: int = 1,
    batch_size: int = 2,
    epochs: int = 1,
    learning_rate: float = 1e-3,
    lr_schedular: str = "cosine_warmup",
    image_mode: str = "L",
    use_zero: bool = False,
    use_amp: bool = False,
    image_size: int = 512,
    val_image_size: int = None,
    val_batch_size: int = None,
    test_image_size: int = None,
    test_batch_size: int = None,
    use_random_resize: bool = False,
    use_random_crop: bool = False,
    use_random_rotation: bool = False,
    color_jitter=0.0,
    center_crop: bool = False,
    center_crop_size: int = 1200,
    center_crop_offset: Tuple[int] = (-60, -50),
    project_name: str = "project_name",
    run_name: str = "run_name",
) -> None:
    """
    :param model_name: Name of model to train
    :param image_folder: Path to images
    :param mask_folder: Path to masks
    :param output_data_dir: Path to output data such as tensorboard logs, argument dump, etc.
    :param tensorboard_dir: Path to tensorboard output location. Will append "/tensorboard" to given path.
    :param model_dir: Directory to save final model weights and the config file.
    :param checkpoint_dir: Directory to save intermediate training checkpoints. If directory not empty, will resume
        training from the 'last.ckpt' checkpoint in the folder.
    :param pretrained_weights: S3 Uri or local path to a .ckpt file. Will load these weights into model prior to train.
        NOTE: If checkpoints exists in checkpoint_dir for model resumption, the weights in checkpoint_dir will
        overwrite the pretrained_weights. This is to enable model resumption on spot pausing.
    :param split_file: Path to json file with keys "train", "val" and "test". Each key has keys of "image" and "mask"
        with lists of filenames for that split. If not given, splits are generated randomly from all available data.
        Can be local path or S3 uri
    :param train_count: Maximum number of training samples to use. Use -1 to use all samples. Otherwise, train_count
        random samples will be used for training.
    :param model_params: Optional dict containing
    :param num_workers: Number or process workers for data loading
    :param batch_size: Batch size per GPU for training
    :param epochs: Number of epochs to train model for
    :param learning_rate: Base learning rate for optimization
    :param lr_schedular: Learning Schedular type, default is cosine with warmup "cosine_warmup"
    :param image_mode: Mode to load images in either RGB or grayscale
    :param use_zero: Uses the ZeroGradient wrapper for efficient multi-gpu optimization
    :param use_amp: If true, use Mixed FP16 precision for training
    :param color_jitter: Float of amount of color jitter to apply as training augmentation
    :param use_random_rotation: If true, applies random rotations as training augmentation
    :param use_random_resize: If true, applies RandomResizedCrop transform as training augmentation
    :param use_random_crop: If true, applies RandomCrop transform as training augmentation.
        If use_random_resize is also True, only applies RandomResizeCrop.
    :param image_size: Images are resized to this prior to being fed to model
    :param val_image_size: Have val loader load image different image size if given
    :param val_batch_size: Optional, batch size for validation loader. Useful is val is using larger/smaller image size
    :param test_image_size: Resizes images to this size for test set if given, otherwise uses val image size
    :param test_batch_size: Optional, batch size for test loader. Useful if test is using larger/smaller image size.
    :param center_crop: If True, takes a center crop of the image and mask prior to transforms
    :param center_crop_size: Size of the center crop
    :param center_crop_offset: Offsets the center of the crop by [Row, Col] or [Height, Width]
    :param project_name: Name of WandB project
    :param run_name: Run name for tracking in WandB project
    """
    logger.info("Starting training run with the following parameters:")
    logger.info(f"{locals()}")

    # Save input information important for inference later as a config.yaml file in the model output dir
    data_config = dict(
        model_name=model_name,
        image_mode=image_mode,
        center_crop=center_crop,
        center_crop_size=center_crop_size,
        center_crop_offset=center_crop_offset,
        image_size=image_size,
        val_image_size=val_image_size,
        test_image_size=test_image_size,
    )
    with open(os.path.join(model_dir, "config.yaml"), "w") as fp:
        yaml.dump(data_config, fp)

    # Log all hyperparameters passed into the model
    logged_config = data_config.update(
        {
            split_file: split_file,
            train_count: train_count,
            # model_params: model_params,
            num_workers: num_workers,
            batch_size: batch_size,
            epochs: epochs,
            learning_rate: learning_rate,
            lr_schedular: lr_schedular,
            use_zero: use_zero,
            use_amp: use_amp,
            val_image_size: val_image_size,
            val_batch_size: val_batch_size,
            test_image_size: test_image_size,
            test_batch_size: test_batch_size,
            use_random_resize: use_random_resize,
            use_random_crop: use_random_crop,
            use_random_rotation: use_random_rotation,
            color_jitter: color_jitter,
        }
    )

    wandb_logger = WandbLogger(
        entity="aqa_llnl", project=project_name, name=run_name, config=logged_config
    )  # , config=vars(args))

    # Get the number of GPUs available for training for use later
    num_gpus = torch.cuda.device_count()

    # Load the Model
    # TODO: Add init params to adjust UNet shape
    # TODO: Add the timm model library to gain access to predefined and pretrained models
    input_channels = 3 if image_mode == "RGB" else 1

    # Create Lightning Module
    module = SegmentationLightningModule(
        model_name=model_name,
        input_channels=input_channels,
        output_channels=1,
        lr=learning_rate,
        use_zero_grad=use_zero and num_gpus > 1,
        schedular_type=lr_schedular,
    )

    logger.info(f"Model: {module.model}")
    logger.info(f"The model has {count_parameters(module.model):,} trainable parameters.")

    if pretrained_weights:
        logger.info(f"Loading model weights from {pretrained_weights}")
        pretrained_weights = download_weights(pretrained_weights)
        checkpoint = torch.load(pretrained_weights)
        module.load_state_dict(checkpoint["state_dict"])

    # Get the data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        split_file=split_file,
        train_count=train_count,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        image_mode=image_mode,
        image_size=image_size,
        val_image_size=val_image_size,
        test_image_size=test_image_size,
        use_random_resize=use_random_resize,
        use_random_rotate=use_random_rotation,
        color_jitter=color_jitter,
        center_crop=center_crop,
        center_crop_size=center_crop_size,
        center_crop_offset=center_crop_offset,
        needs_boxes=module.model.needs_boxes,
    )

    # Create callbacks
    model_ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        mode="min",
        filename="{epoch:02d}-{val_loss:.3f}",
        every_n_epochs=1,
        save_last=True,  # Always saves last chkpt as last.ckpt
        save_on_train_epoch_end=True,
        save_top_k=5,  # Ensures all models are saved
        verbose=True,
    )
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        model_ckpt_callback,
    ]

    # Create loggers
    loggers = [TensorBoardLogger(tensorboard_dir, name="tensorboard")]

    # Create Training Strategy
    if num_gpus > 0:
        strategy = "ddp" if num_gpus > 1 else "auto"
        accelerator = "gpu"
    else:
        strategy = "auto"
        accelerator = "cpu"

    # See if prior run exists
    resume_checkpoint = None
    if os.path.exists(os.path.join(checkpoint_dir, "last.ckpt")):
        resume_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")
        if pretrained_weights:
            logger.warning(
                f"Overwriting provided pretrained weights from {pretrained_weights} with "
                f"training checkpoint {resume_checkpoint}"
            )

    # Create Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=max(num_gpus, 1),
        strategy=strategy,
        default_root_dir=output_data_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        precision="16-mixed" if use_amp else 32,
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_checkpoint)

    # Shut down process group if launched to ensure training doesn't hang
    if strategy == "ddp":
        torch.distributed.destroy_process_group()
    # Run Testing on a single device
    if trainer.is_global_zero:
        tester = pl.Trainer(
            logger=wandb_logger,
            devices=1,
            num_nodes=1,
            max_epochs=epochs,
            accelerator=accelerator,
            callbacks=callbacks,
        )

        # Load the best model checkpoint
        logger.info(f"Testing with best performing model parameters: \n\t{model_ckpt_callback.best_model_path}")
        # Run test using the best model weights path
        test_module = SegmentationLightningModule.load_from_checkpoint(model_ckpt_callback.best_model_path)
        test_results = tester.test(model=test_module, dataloaders=test_loader)
        tester.save_checkpoint(os.path.join(model_dir, "best.ckpt"), weights_only=True)

        # Save the images here?
        logger.info(f"Saving {len(test_module.test_log_images)}")
        for idx, (img, gt_mask, pred_mask) in enumerate(test_module.test_log_images_raw):
            save_test_result_image(img, gt_mask, pred_mask, os.path.join(output_data_dir, f"test_{idx}.png"))

        # Try saving to wandb
        logger.info("Saving images to wandb")
        wandb_logger.experiment.log({"test_images": test_module.test_log_images})

        # Save the test results in a yaml file in the output directory
        with open(os.path.join(output_data_dir, "test_results.yaml"), "w") as fp:
            yaml.dump(test_results, fp)


def save_test_result_image(image, gt_mask, pred_mask, fname):
    fig_len = image.shape[0] / 100
    fig, axs = plt.subplots(1, 4, figsize=(fig_len * 4, fig_len))
    # Add titles
    axs[0].set_title("Input Image")
    axs[1].set_title("Mask")
    axs[2].set_title("Raw Output")
    axs[3].set_title("Thresholded Output")
    axs[0].imshow(image, cmap="gray")
    axs[0].set_axis_off()
    axs[1].imshow(gt_mask, cmap="gray")
    axs[1].set_axis_off()
    axs[2].imshow(pred_mask)
    axs[2].set_axis_off()
    axs[3].imshow(pred_mask > 0.5, cmap="gray")
    axs[3].set_axis_off()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args, model_args = parse_args()
    kwargs = vars(args)
    kwargs["model_params"] = model_args
    main(**kwargs)
