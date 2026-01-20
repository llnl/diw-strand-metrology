import argparse
import logging
import os

import boto3
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pathlib import Path
from matplotlib import pyplot as plt
import mlflow

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from typing import Optional, Tuple

from llnl_ml.model import count_parameters
from llnl_ml.model.util import get_training_data_sources, get_sagemaker_resource_config
from llnl_ml.model.builder import get_model_class
from llnl_ml.data.builder import get_dataloaders
from llnl_ml.lightning import SegmentationLightningModule
from llnl_ml.util import str2bool, str2intlist
from llnl_ml.mlflow_dataset import log_dataset_to_mlflow

from pytorch_lightning.loggers import MLFlowLogger


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
    parser.add_argument("--accumulate_iters", type=int, help="Accumulates gradients over N iterations for larger effective batch size", default=10)
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument("--lr_schedular", type=str, default="cosine_warmup", help="LR Schedular")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Number of epochs to apply linear warmup lr schedule.")
    parser.add_argument(
        "--image_mode",
        type=str,
        choices=["RGB", "grayscale"],
        default="grayscale",
        help="Image mode for reading images (RGB or grayscale)",
    )
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision to use for model.")
    parser.add_argument(
        "--project_name",
        type=str,
        default="project_name2",
        help="Name of the MLFlow experiment for organizing related runs",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="run_name2",
        help="Name of the specific MLFlow run for tracking this training execution",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="",
        help=(
            "Filepath or S3Uri to metadata csv file containing image names and metadata fields. "
            "If provided, will aggregate results by fields in csv, output to logs, and generate pandas dataframe."
        ),
    )
    parser.add_argument(
        "--transform_config",
        type=str,
        default="",
        help=(
            "Path to YAML configuration file for data transforms. "
            "If not provided, will use default transforms from src/llnl_ml/configs/default_transforms.yaml"
        ),
    )
    parser.add_argument(
        "--max_hausdorff_size", 
        type=int, 
        default=256, 
        help=(
            "Maximum allowed image size for computing Hausdorff Distance. "
            "Images larger than max, resized to max. Use -1 for no limit."
        )
    )

    args, model_params = parser.parse_known_args()
    
    # Check for deprecated transform arguments and raise error
    deprecated_args = [
        'use_random_resize', 'use_random_crop', 'use_random_rotation', 
        'color_jitter', 'image_size', 'val_image_size', 'test_image_size',
        'center_crop', 'center_crop_size', 'center_crop_offset'
    ]
    
    used_deprecated = []
    for arg in deprecated_args:
        if hasattr(model_params, arg):
            used_deprecated.append(f"--{arg}")
    
    if used_deprecated:
        raise ValueError(
            f"The following CLI arguments are deprecated: {', '.join(used_deprecated)}. "
            f"Please use --transform_config to specify a YAML configuration file instead. "
            f"See src/llnl_ml/configs/default_transforms.yaml for an example."
        )
    
    return args, model_params


def download_s3_uri(s3_uri: str) -> str:
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
    transform_config: str = "",
    train_count: int = -1,
    model_params: Optional[dict] = None,
    num_workers: int = 1,
    batch_size: int = 2,
    accumulate_iters: int = 1,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    lr_schedular: str = "cosine_warmup",
    warmup_epochs: int = 1,
    image_mode: str = "L",
    precision: str = "16-mixed",
    val_batch_size: Optional[int] = None,
    test_batch_size: int = 1,
    project_name: str = "project_name",
    run_name: str = "run_name",
    metadata_file: str = "",
    max_hausdorff_size: int = 512,
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
    :param transform_config: Path to YAML configuration file for data transforms. If not provided, uses default config.
    :param train_count: Maximum number of training samples to use. Use -1 to use all samples. Otherwise, train_count
        random samples will be used for training.
    :param model_params: Optional dict containing model-specific parameters
    :param num_workers: Number or process workers for data loading
    :param batch_size: Batch size per GPU for training
    :param accumulate_iters: Number of iters to accumlate gradients over increasing effective batch size.
    :param epochs: Number of epochs to train model for
    :param learning_rate: Base learning rate for optimization
    :param lr_schedular: Learning Schedular type, default is cosine with warmup "cosine_warmup"
    :param image_mode: Mode to load images in either RGB or grayscale
    :param precision: Precision value to use with lightning trainer. Default 16-mixed.
    :param val_batch_size: Optional, batch size for validation loader. Useful is val is using larger/smaller image size
    :param test_batch_size: Optional, batch size for test loader. Useful if test is using larger/smaller image size.
    :param project_name: Name of the MLFlow experiment for organizing related runs
    :param run_name: Name of the specific MLFlow run for tracking this training execution
    :param metadata_file: Filepath or S3Uri to metadata csv file.
    """
    # Determine if this is the main process (for logging)
    # In non-distributed mode, this will always be True
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
    
    if is_main_process:
        logger.info("Starting training run with the following parameters:")
        logger.info(f"{locals()}")

    # Get training data sources (S3 URIs) using hybrid approach
    training_data_info = get_training_data_sources()
    resource_config = get_sagemaker_resource_config()
    
    if is_main_process and resource_config:
        current_host = resource_config.get('current_host', 'unknown')
        hosts = resource_config.get('hosts', [])
        logger.info(f"SageMaker Resource Config: Host {current_host} of {hosts}")

    # Save input information important for inference later as a config.yaml file in the model output dir
    data_config = dict(
        model_name=model_name,
        image_mode=image_mode,
        transform_config=transform_config,
    )
    with open(os.path.join(model_dir, "config.yaml"), "w") as fp:
        yaml.dump(data_config, fp)

    # Ensure output directory exists for MLFlow tracking
    os.makedirs(output_data_dir, exist_ok=True)

    # Log all hyperparameters passed into the model
    logged_config = {
        "split_file": split_file,
        "transform_config": transform_config,
        "train_count": train_count,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lr_schedular": lr_schedular,
        "precision": precision,
        "val_batch_size": val_batch_size,
        "test_batch_size": test_batch_size,
    }
    logged_config.update(data_config)
    
    # Add training data source metadata to logged config for MLFlow tracking
    if training_data_info.get('input_data_config'):
        # Add SageMaker job metadata if available
        if training_data_info.get('training_job_name'):
            logged_config['sagemaker_training_job_name'] = training_data_info['training_job_name']
        if training_data_info.get('training_job_arn'):
            logged_config['sagemaker_training_job_arn'] = training_data_info['training_job_arn']
        if training_data_info.get('instance_type'):
            logged_config['sagemaker_instance_type'] = training_data_info['instance_type']
        if training_data_info.get('instance_count'):
            logged_config['sagemaker_instance_count'] = training_data_info['instance_count']
    
    if resource_config:
        logged_config['sagemaker_current_host'] = resource_config.get('current_host', 'unknown')
        logged_config['sagemaker_hosts'] = resource_config.get('hosts', [])

    # Get the number of GPUs available for training for use later
    num_gpus = torch.cuda.device_count()
    
    # Initialize MLFlow logger with proper error handling
    # For DDP, delay initialization until after strategy setup to avoid serialization issues
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_logger = MLFlowLogger(
        experiment_name=project_name,
        run_name=run_name,
        tracking_uri=mlflow_tracking_uri,
        save_dir=f"{output_data_dir}/mlruns",
        log_model=True,
    )

    if is_main_process:
        logger.info(f"MLFlow Tracking URI: {mlflow_tracking_uri}")
        logger.info(f"[-] {mlflow_logger._experiment_name} - {mlflow_logger._run_name} - {mlflow_logger.run_id}")
        logger.info(f"Number of GPUs detected: {num_gpus}")

    # Store config for later logging
    hyperparams_to_log = logged_config

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
        schedular_type=lr_schedular,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        max_hausdorff_size=max_hausdorff_size,
    )

    if is_main_process:
        logger.info(f"Model: {module.model}")
        logger.info(f"The model has {count_parameters(module.model):,} trainable parameters.")

    if pretrained_weights:
        if is_main_process:
            logger.info(f"Loading model weights from {pretrained_weights}")
        pretrained_weights = download_s3_uri(pretrained_weights)
        checkpoint = torch.load(pretrained_weights)
        module.load_state_dict(checkpoint["state_dict"])

    # Get the model class to determine dataset requirements
    try:
        model_class = get_model_class(model_name)
        logger.info(f"Using model class: {model_class.__name__}")
    except ValueError as e:
        logger.warning(f"Could not get model class: {e}. Using default dataset format.")
        model_class = None

    # Get the data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        image_folder=image_folder,
        mask_folder=mask_folder,
        batch_size=batch_size,
        split_file=split_file,
        transform_config=transform_config,
        train_count=train_count,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        model_class=model_class,
    )

    # Log dataset information to MLFlow
    # This is done after data loaders are created to enable fallback split extraction
    if is_main_process:
        dataset_logged = log_dataset_to_mlflow(
            training_data_info=training_data_info,
            split_file=split_file,
            mlflow_logger=mlflow_logger,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            image_folder=image_folder,
            mask_folder=mask_folder
        )
        if dataset_logged:
            logger.info("Dataset tracking completed successfully")
        else:
            logger.info("Dataset tracking was skipped or failed - training will continue")

    # Load in metadata if available
    metadata = load_metadata(metadata_file)

    # Create callbacks
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_total_loss",
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

    # Create Training Strategy
    # Use DDPStrategy with find_unused_parameters=False to reduce memory overhead
    if num_gpus > 0:
        strategy = "ddp" if num_gpus > 1 else "auto"
        accelerator = "gpu"
    else:
        strategy = "auto"
        accelerator = "cpu"

    if is_main_process:
        logger.info(f"Training strategy: {strategy}")
        logger.info(f"Accelerator: {accelerator}")

    # See if prior run exists
    resume_checkpoint = None
    if os.path.exists(os.path.join(checkpoint_dir, "last.ckpt")):
        resume_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")
        if pretrained_weights and is_main_process:
            logger.warning(
                f"Overwriting provided pretrained weights from {pretrained_weights} with "
                f"training checkpoint {resume_checkpoint}"
            )

    # Warn about gradient accumulation instability
    if accumulate_iters > 1:
        logger.warning(
            f"WARNING: accumulate_grad_batches={accumulate_iters} > 1 is currently unstable "
            f"with newer PyTorch versions and may cause training to crash or stall. "
            f"Consider using a larger batch_size instead."
        )

    # Create Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=max(num_gpus, 1),
        strategy=strategy,
        default_root_dir=output_data_dir,
        logger=mlflow_logger,
        callbacks=callbacks,
        precision=precision,
        log_every_n_steps=10,
        accumulate_grad_batches=accumulate_iters,
    )

    # Train the model
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_checkpoint)

    # Shut down DDP process group before single-device testing to prevent hanging
    # This is necessary because we're switching from distributed to single-device execution
    if strategy == "ddp" and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    # Run testing on a single device to avoid DDP issues with uneven batch distribution
    # Only rank 0 performs testing to ensure all samples are evaluated exactly once
    if trainer.is_global_zero:
        # Create a single-device trainer for testing
        tester = pl.Trainer(
            logger=mlflow_logger,
            devices=1,
            num_nodes=1,
            accelerator=accelerator,
            callbacks=callbacks,
        )

        # Load the best model checkpoint
        if is_main_process:
            logger.info(f"Testing with best performing model parameters: \n\t{model_ckpt_callback.best_model_path}")
        
        # Run test using the best model weights path
        test_module = SegmentationLightningModule.load_from_checkpoint(model_ckpt_callback.best_model_path)
        test_results = tester.test(model=test_module, dataloaders=test_loader)
        tester.save_checkpoint(os.path.join(model_dir, "best.ckpt"), weights_only=True)

        # Save the images to the output data directory
        # logger.info(f"Saving {len(best_module.test_log_images)}")
        # for idx, (img, gt_mask, pred_mask) in enumerate(best_module.test_log_images_raw):
        #     save_test_result_image(img, gt_mask, pred_mask, os.path.join(output_data_dir, f"test_{idx}.png"))

        # Save the images to MLFlow directly
        logger.info("Saving images to MLFlow")
        try:
            # Create test_images directory in output_data_dir
            test_images_dir = os.path.join(output_data_dir, "test_images")
            os.makedirs(test_images_dir, exist_ok=True)
            
            # Iterate through test_log_images and save each as PNG
            for idx, img_data in enumerate(test_module.test_log_images):
                img_path = os.path.join(test_images_dir, f"test_image_{idx}.png")
                create_test_image_visualization(img_data, img_path)
            
            # Log the test_images directory as MLFlow artifacts
            mlflow_logger.experiment.log_artifacts(
                run_id=mlflow_logger.run_id,
                local_dir=test_images_dir,
                artifact_path="test_images"
            )
            logger.info(f"Successfully logged {len(test_module.test_log_images)} test images to MLFlow")
        except Exception as e:
            logger.exception(f"Failed to log test images to MLFlow: {e}")

        # Since testing runs on single device, no need to gather metrics from multiple ranks
        per_image_metrics = pd.DataFrame(test_module.test_image_metrics)
        # If metadata was provided, append the metadata to the output metrics
        # and generate per metadata results
        if metadata is not None:
            # Join the metadata dataframe to the per image metrics
            per_image_metrics = pd.merge(per_image_metrics, metadata, on="image_name", how="left")
            # Get the metadata column names
            meta_columns = set(metadata.columns) - {"S3 URI", "image_name"}
            # For each column
            for column in meta_columns:
                column_group = column if column_is_discrete(metadata, column) else pd.cut(metadata[column], bins=10)
                grouped_mean = per_image_metrics.groupby(column_group)[["acc", "dice", "jaccard"]].mean()
                # Log grouped metrics to MLFlow
                for col in ["acc", "dice", "jaccard"]:
                    metrics_dict = {}
                    for idx in grouped_mean.index:
                        metric_name = f"test_{col}_{column}_{idx}"
                        metric_value = float(grouped_mean.loc[idx, col])
                        metrics_dict[metric_name] = metric_value
                    mlflow_logger.log_metrics(metrics_dict)

        # Save per_image_metrics locally as parquet

        per_image_metrics_path = os.path.join(output_data_dir, "per_image_metrics.csv")
        per_image_metrics.to_csv(per_image_metrics_path, index=False)
        
        # Log per_image_metrics to MLFlow as artifact
        try:
            mlflow_logger.experiment.log_artifact(
                run_id=mlflow_logger.run_id,
                local_path=per_image_metrics_path,
                artifact_path="metrics/per_image_metrics.csv"
            )
            logger.info("Successfully logged per_image_metrics.parquet to MLFlow")
        except Exception as e:
            logger.exception(f"Failed to log per_image_metrics to MLFlow: {e}")

        # Save the test results in a yaml file in the output directory
        test_results_path = os.path.join(output_data_dir, "test_results.yaml")
        with open(test_results_path, "w") as fp:
            yaml.dump(test_results, fp)
        
        # Log test results to MLFlow as artifact
        try:
            mlflow_logger.experiment.log_artifact(
                run_id=mlflow_logger.run_id,
                local_path=test_results_path,
                artifact_path="metrics"
            )
            logger.info("Successfully logged test_results.yaml to MLFlow")
        except Exception as e:
            logger.exception(f"Failed to log test_results to MLFlow: {e}")


def load_metadata(metadata_file: str) -> Optional[pd.DataFrame]:
    """
    Loads the given metadata file from the path and returns it as a pandas dataframe.

    :param metadata_file: Local file path or s3 uri to metadata file (csv or parquet format)
    :return:
        DataFrame containing the metadata
    """
    if not metadata_file:
        return None

    metadata_file = download_s3_uri(metadata_file)
    if metadata_file.endswith("csv"):
        metadata = pd.read_csv(metadata_file)
    elif metadata_file.endswith("parquet"):
        metadata = pd.read_parquet(metadata_file)
    else:
        raise ValueError(
            f"Expect metadata to be either CSV or Parquet format. "
            f"File given ends with {os.path.splitext(metadata_file)[-1]}"
        )

    # Add the image basename as column to the dataframe for easier reference
    s3_uris = metadata["S3 URI"]
    image_names = [os.path.basename(uri).strip() for uri in s3_uris]

    metadata["image_name"] = image_names

    return metadata


def column_is_discrete(dataframe: pd.DataFrame, column: str, heuristic_percent: float = 0.8) -> bool:
    """
    Heuristic to determine if the column contains continuous or discrete values
    :param dataframe: DataFrame with desired data
    :param column: Name of the column to test
    :param heuristic_percent: If this percent of values are unique, consider the column continuous
    :return: Returns True if column appears to be discrete, False if continuous
    """
    if dataframe[column].dtype == object:
        return True
    return (dataframe[column].nunique() / len(dataframe)) < heuristic_percent


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


def create_test_image_visualization(img_data: dict, output_path: str) -> None:
    """
    Create a composite visualization of test image with ground truth and prediction.
    
    :param img_data: Dictionary containing 'image' (PIL Image), 'pred_mask' (numpy array), 
                     and 'gt_mask' (numpy array)
    :param output_path: Path where the visualization should be saved
    """
    import numpy as np
    
    # Extract data from dictionary
    pil_image = img_data["image"]
    pred_mask = img_data["pred_mask"]
    gt_mask = img_data["gt_mask"]
    
    # Convert PIL image to numpy array for visualization
    img_array = np.array(pil_image)
    
    # Create figure with 3 subplots: input, ground truth, prediction
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(img_array, cmap="gray")
    axs[0].set_title("Input")
    axs[0].axis("off")
    
    axs[1].imshow(gt_mask, cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")
    
    axs[2].imshow(pred_mask, cmap="gray")
    axs[2].set_title("Prediction")
    axs[2].axis("off")
    
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args, model_args = parse_args()
    kwargs = vars(args)
    kwargs["model_params"] = model_args
    main(**kwargs)
