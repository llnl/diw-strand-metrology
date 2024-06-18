# Builds on the SageMaker provided PyTorch 2.2 Training Image
# Base DockerFile content can be found here:
# https://github.com/aws/deep-learning-containers/blob/master/pytorch/training/docker/2.2/py3/cu121/Dockerfile.gpu
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.2-gpu-py310

# Copy the requirements file into container
COPY src/requirements.txt requirements.txt

# Install the python requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

# Copy final steps of the docker file to clear cache and set entrypoint
# Removing the cache as it is needed for security verification
RUN rm -rf /root/.cache | true

ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]
CMD ["/bin/bash"]
