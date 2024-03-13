import argparse
import boto3
import json
import logging
import os
import sys

from sklearn.model_selection import train_test_split
from typing import Sequence


logger = logging.getLogger("CreateDataSplit")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_s3_files(s3_uri: str, region: str = "us-west-2") -> Sequence[str]:
    s3 = boto3.resource("s3")
    # Split s3 uri into bucket and prefix
    bucket, prefix = s3_uri.removeprefix("s3://").split("/", 1)
    # Add ending "/" if not present on prefix
    prefix = prefix if prefix.endswith("/") else prefix + "/"

    # Get filenames for all objects on the prefix
    filenames = []
    for obj in s3.Bucket(bucket).objects.filter(Prefix=prefix):
        filename = obj.key.removeprefix(prefix)
        if filename.endswith(IMAGE_EXTENSIONS):
            filenames.append(filename)

    return sorted(filenames)


def list_local_files(local_path: str) -> Sequence[str]:
    filenames = [fname for fname in os.listdir(local_path) if fname.endswith(IMAGE_EXTENSIONS)]

    return sorted(filenames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", help="Local path or S3 URI to images", type=str)
    parser.add_argument("--mask_path", help="Local path or S3 URI to masks", type=str)
    parser.add_argument("--split_file", help="Local path or S3 URI to store generated split file", type=str)
    parser.add_argument(
        "--train_percent", help="Percent of data used for training. Default 0.8", type=float, default=0.8
    )
    parser.add_argument(
        "--val_percent", help="Percent of data used for validation. Default 0.1", type=float, default=0.1
    )
    parser.add_argument("--test_percent", help="Percent of data used for test. Default 0.1", type=float, default=0.1)
    parser.add_argument(
        "--region", help="S3 region bucket is located in. Default us-west-2", type=str, default="us-west-2"
    )
    parser.add_argument("--seed", help="Integer seed for randomizer", type=int, default=42)

    return parser.parse_args()


def main(
    image_path: str,
    mask_path: str,
    split_file: str,
    train_percent: float = 0.8,
    val_percent: float = 0.1,
    test_percent: float = 0.1,
    seed: int = 42,
    region: str = "us-west-2",
) -> None:
    # Get list of image and mask filenames
    logger.info("Finding images files...")
    if image_path.startswith("s3://"):
        image_filenames = list_s3_files(image_path, region)
    else:
        image_filenames = list_local_files(image_path)

    if mask_path.startswith("s3://"):
        mask_filenames = list_s3_files(mask_path, region)
    else:
        mask_filenames = list_local_files(mask_path)

    # Ensure image and mask have same length
    assert len(image_filenames) == len(mask_filenames), "The image and mask folders have different number of images"

    logger.info(f"Found {len(image_filenames)} images and masks")

    # Check that the splits sum to 1
    if not (train_percent + val_percent + test_percent) == 1.0:
        logger.warning(
            "The three splits provided do not sum to 1.0. Given:"
            f"\n\ttraing_percent: {train_percent}, val_percent: {val_percent}, test_percent: {test_percent}"
            "\nNormalizing values to sum to 1.0"
        )
        total = train_percent + val_percent + test_percent
        train_percent = train_percent / total
        val_percent = val_percent / total
        test_percent = test_percent / total
        logger.warning(
            "Splitting data using following percents:"
            f"\n\ttraing_percent: {train_percent:.2f}, val_percent: {val_percent:.2f}, test_percent: {test_percent:.2f}"
        )

    img_train, img_temp, mask_train, mask_temp = train_test_split(
        image_filenames, mask_filenames, train_size=train_percent, random_state=seed
    )

    val_percent = val_percent / (val_percent + test_percent)
    img_val, img_test, mask_val, mask_test = train_test_split(
        img_temp, mask_temp, train_size=val_percent, random_state=seed
    )

    split_dict = {
        "train": {"image": img_train, "mask": mask_train},
        "val": {"image": img_val, "mask": mask_val},
        "test": {"image": img_test, "mask": mask_test},
    }
    logger.info(f"Created splits of size:\n\ttrain: {len(img_train)}, val: {len(img_val)}, test: {len(img_test)}")

    # Dump the dict to a string
    split_string = json.dumps(split_dict)

    # If split_file is s3 path, push to S3
    if split_file.startswith("s3://"):
        s3 = boto3.resource("s3", region)
        bucket, key = split_file.removeprefix("s3://").split("/", 1)
        s3_obj = s3.Object(bucket, key)
        resp = s3_obj.put(Body=split_string.encode(), ContentType="application/json")
        if resp.get("ResponseMetadata", {}).get("HTTPStatusCode") != 200:
            raise ValueError(
                f"An error occurred while uploading file to S3. Received response:\n {json.dumps(resp, indent=2)}"
            )
    # Otherwise, save locally
    else:
        # Ensure directory exists
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        with open(split_file, "w") as fp:
            fp.write(split_string)


if __name__ == "__main__":
    args = parse_args()
    kwargs = vars(args)
    main(**kwargs)
