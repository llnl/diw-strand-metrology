import json
import logging
import os
from torch import nn

logger = logging.getLogger(__name__)


def he_initialization(layer):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_sagemaker_training_job_info():
    """
    Query SageMaker API to get training job details including S3 input URIs.
    
    This is the most reliable method to get the original S3 URIs used for training,
    as the inputdataconfig.json file doesn't contain this information.
    
    Returns:
        dict: Training job information including input data config with S3 URIs.
              Returns empty dict if not running on SageMaker or if query fails.
              
    Example return value:
        {
            'training_job_name': 'my-training-job-2024-01-01-12-00-00',
            'training_job_arn': 'arn:aws:sagemaker:...',
            'input_data_config': {
                'train_image': {
                    'S3Uri': 's3://bucket/path/to/images',
                    'InputMode': 'FastFile',
                    'ContentType': 'application/x-image'
                },
                'train_mask': {
                    'S3Uri': 's3://bucket/path/to/masks',
                    'InputMode': 'FastFile',
                    'ContentType': 'application/x-image'
                }
            },
            'instance_type': 'ml.p4d.24xlarge',
            'instance_count': 1,
            'source': 'sagemaker_api'
        }
    """
    training_job_name = os.environ.get('TRAINING_JOB_NAME')
    
    if not training_job_name:
        logger.debug("TRAINING_JOB_NAME environment variable not found. Not running on SageMaker.")
        return {}
    
    try:
        import boto3
        
        sagemaker_client = boto3.client('sagemaker')
        response = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        
        # Extract relevant information
        job_info = {
            'training_job_name': training_job_name,
            'training_job_arn': response.get('TrainingJobArn'),
            'input_data_config': {},
            'instance_type': response.get('ResourceConfig', {}).get('InstanceType'),
            'instance_count': response.get('ResourceConfig', {}).get('InstanceCount'),
            'source': 'sagemaker_api'
        }
        
        # Extract S3 URIs from input data config
        for channel in response.get('InputDataConfig', []):
            channel_name = channel['ChannelName']
            s3_data_source = channel.get('DataSource', {}).get('S3DataSource', {})
            
            job_info['input_data_config'][channel_name] = {
                'S3Uri': s3_data_source.get('S3Uri'),
                'InputMode': channel.get('InputMode'),
                'ContentType': channel.get('ContentType'),
                'S3DataType': s3_data_source.get('S3DataType'),
                'S3DataDistributionType': s3_data_source.get('S3DataDistributionType'),
            }
        
        logger.info(f"Successfully retrieved training job info for: {training_job_name}")
        return job_info
        
    except ImportError:
        logger.warning("boto3 not available. Cannot query SageMaker API.")
        return {}
    except Exception as e:
        logger.warning(f"Failed to describe training job '{training_job_name}': {e}")
        return {}


def get_training_data_sources():
    """
    Get training data S3 URIs using multiple fallback methods.
    
    This function tries multiple methods to retrieve the S3 URIs for training data:
    1. Query SageMaker training job API (most reliable, requires IAM permissions)
    2. Read from environment variables (manual override capability)
    3. Return empty dict (local development)
    
    Environment variable format for manual override:
        TRAIN_IMAGE_S3_URI=s3://bucket/path/to/images
        TRAIN_MASK_S3_URI=s3://bucket/path/to/masks
    
    Returns:
        dict: Training data source information with the following structure:
            {
                'input_data_config': {
                    'train_image': {'S3Uri': 's3://...', ...},
                    'train_mask': {'S3Uri': 's3://...', ...}
                },
                'training_job_name': '...',  # If from SageMaker API
                'instance_type': '...',       # If from SageMaker API
                'source': 'sagemaker_api' | 'environment' | 'local'
            }
    """
    # Method 1: Try SageMaker API first (most reliable)
    job_info = get_sagemaker_training_job_info()
    if job_info and job_info.get('input_data_config'):
        logger.info("Retrieved training data sources from SageMaker API")
        return job_info
    
    # Method 2: Fall back to environment variables (manual override)
    env_s3_uris = {}
    for key, value in os.environ.items():
        if key.endswith('_S3_URI'):
            # Convert TRAIN_IMAGE_S3_URI -> train_image
            channel_name = key.replace('_S3_URI', '').lower()
            env_s3_uris[channel_name] = {'S3Uri': value}
    
    if env_s3_uris:
        logger.info(f"Retrieved training data sources from environment variables: {list(env_s3_uris.keys())}")
        return {
            'input_data_config': env_s3_uris,
            'source': 'environment'
        }
    
    # Method 3: Running locally or no data source info available
    logger.debug("No training data source information available (running locally)")
    return {'source': 'local'}


def get_sagemaker_resource_config():
    """
    Read SageMaker resource configuration to get training job metadata.
    
    Returns:
        dict: Resource configuration including current host, hosts list, etc.
              Returns empty dict if running locally or config not found.
    """
    config_path = "/opt/ml/input/config/resourceconfig.json"
    
    try:
        with open(config_path, 'r') as f:
            resource_config = json.load(f)
        return resource_config
    except FileNotFoundError:
        logger.debug(f"Resource config not found at {config_path}. Running locally or outside SageMaker.")
        return {}
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse resource config: {e}")
        return {}


def register_dataset():
    pass