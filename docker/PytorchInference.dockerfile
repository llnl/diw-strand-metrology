# Set device to cpu or gpu based on desired inference deployment instance type
ARG DEVICE=cpu

# Builds on the SageMaker provided PyTorch 2.2 Training Image
# Base DockerFile content can be found here:
# CPU:https://github.com/aws/deep-learning-containers/blob/master/pytorch/inference/docker/2.2/py3/Dockerfile.cpu
# GPU: https://github.com/aws/deep-learning-containers/blob/master/pytorch/inference/docker/2.2/py3/cu118/Dockerfile.gpu
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.2-${DEVICE}-py310

# Copy the requirements file into container
COPY src/requirements.txt requirements.txt

# Install the python requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

# Copy final steps of the docker file to clear cache and set entrypoint
# Removing the cache as it is needed for security verification
RUN rm -rf /root/.cache | true

# Set Entrypoint for torch serve
EXPOSE 8080 8081
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]
CMD ["torchserve", "--start", "--ts-config", "/home/model-server/config.properties", "--model-store", "/home/model-server/"]