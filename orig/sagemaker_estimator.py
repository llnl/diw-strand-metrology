from sagemaker.experiments import Run
from sagemaker.pytorch import PyTorch
from sagemaker import Session
from sagemaker.utils import unique_name_from_base

train_image = "s3://<BUCKET>/images"
train_mask = "s3://<BUCKET>/masks"

experiment_name = "unet-segmentation"
run_name = "run1-segmentation"

with Run(experiment_name=experiment_name, run_name=run_name, sagemaker_session=Session()) as run:
    estimator = PyTorch(
        entry_point="unet_segmenter.py",
        source_dir=".",
        role="<ROLE>",
        framework_version="1.13.1",
        py_version="py39",
        instance_count=1,
        instance_type= 'ml.p3dn.24xlarge', #'ml.p3.8xlarge', #'ml.p4de.24xlarge', # 80GB A100 GPU   "ml.g4dn.4xlarge", 
        hyperparameters={
            "num_workers": 32,
            "batch_size": 8, #16, #32 #
            "epochs": 1,
            "learning_rate": 0.001,
            "image_mode": "grayscale",
            "use_zero":"True", # Note comment on bool as type. Empty String => "" = False; Any string => "Hello" | "True" | "Flase" = True; https://docs.python.org/3/library/argparse.html#type
            "use_amp": "True", # See above comment
        }, 
    )

estimator.fit(
    {"train_image": train_image, "train_mask": train_mask}
)