# Standalone SageMaker Container Dockerfile
# This Dockerfile packages a trained SageMaker model into a self-contained container
# that can be deployed independently of AWS SageMaker infrastructure.

# Build Arguments
# MODEL_PATH: Path to the untarred model archive directory (relative or absolute)
# INFERENCE_IMAGE: Base SageMaker inference image (e.g., 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.2-cpu-py310)
ARG MODEL_PATH
ARG INFERENCE_IMAGE

# Extend from parameterized SageMaker inference base image
FROM ${INFERENCE_IMAGE}

# Install Python dependencies from model archive
# Copy requirements first for better Docker layer caching
COPY ${MODEL_PATH}/code/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy model artifacts to standard SageMaker locations
# Model checkpoints (best.ckpt or last.ckpt)
COPY ${MODEL_PATH} /opt/ml/model

# Copy inference code to SageMaker code directory
COPY ${MODEL_PATH}/code/ /opt/ml/code/

# Configure SageMaker environment variables
# These tell the SageMaker serving infrastructure where to find the inference code
ENV SAGEMAKER_PROGRAM=inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# Set the model directory environment variable
# This is used by model_fn to locate the model artifacts
ENV SAGEMAKER_MODEL_DIR=/opt/ml/model

# Add the code directory to PYTHONPATH so inference.py and llnl_ml can be imported
ENV PYTHONPATH=/opt/ml/code:${PYTHONPATH}
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_MODEL_DIR=/opt/ml/model
ENV TS_MAX_REQUEST_SIZE=26214400

# Expose ports for inference and management
EXPOSE 8080 8081

# Override the default CMD to use Python serving mode
# When SAGEMAKER_PROGRAM is set, dockerd-entrypoint.py uses the Python serving stack
# instead of TorchServe, which calls model_fn, input_fn, predict_fn, output_fn
CMD ["serve"]
