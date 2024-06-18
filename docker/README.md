# SageMaker Docker Containers
Instructions for building and utilizing your own docker containers for SageMaker training and inference.
The docker files provided extend the containers managed and provided by SageMaker to included all the requirements for this repo.
There are a few reasons to go this route when using SageMaker Training and Inference:
* Locks in the package versions for consistent comparisons during training.
* Prevents updates from the SageMaker team breaking your deployments.
* Faster container deployment as all packages already installed and no longer need to be installed from pypi.
* Ability to train/deploy models in situations where full internet access is restricted.

However, this does put more responsibility on the account holders to maintain these containers.
We recommend rebuilding these containers periodically.
Containers should always be updated in the following situations:
* Resolve CVEs in any container dependency (OS up to python packages)
* Updates to specific python packages for feature dependencies.


## Prerequisites
You will require a machine with the following:
* Docker installed
* At least 30GB of free harddrive space (recommend 50GB)
* This repository available locally

If your normal development environment does not meet these requirements, you can use an EC2 instance.
We provide instructions for setting up an EC2 instance in the next section.
If your normal development machine is set up, you can skip to [Build the Docker Containers](#build-the-docker-containers) section.

If there are concerns with utilizing Docker, we recommend [Finch](https://runfinch.com/) as a fully opensource alternative
Finch is available on Windows (via WSL2) or MacOS and works as a drop in replacement of Docker.
If using finch, just replace any `docker` command with `finch`. For example

```shell
docker build .
# becomes
finch build .
```

## Setting up EC2 Instance
Before starting, you will need to ensure you have access to or the ability to create an EC2 IAM Role with the following minimum permissions.
Below each permission is a recommended policy to grant the desired permissions.
These policies can also be used as a template to further restrict access by limiting the specific resources to grant access to.
* Push/Pull access to ECR
  * `AmazonEC2ContainerRegistryPowerUser` policy
* Clone access to CodeCommit
  * `AWSCodeCommitReadOnly`
* SSM Management Access
  * `AmazonSSMManagedInstanceCore`

### Create and launch instance
1. Navigate to the EC2 console page in your AWS Account
2. Click the **"Launch Instance"** button. This is available in the EC2 dashboard, or under the **"Instances"** section in the left navigation pane.
3. Name the instance as desired, for example `Docker-builder`
4. In the **Application and OS Images** section, select **"Amazon Linux"** under the **Quick Start** section. In the AMI dropdown below, ensure _Amazon Linux 20203 AMI_ is selected. This will likely be the default option.
5. For **Instance Type**, we recommend an `m6a.2xlarge` or an `m5.2xlarge` instance type.
6. For **Key Pair**, we recommend selecting "Proceed without a key pair" as we will utilize the SSM Connection. If you prefer ssh access, create a new key pair for this instance.
7. Under **Network settings** we will modify the following
   * Use the default VPC, unless policy requires the use of a different VPC
   * Select **Create security group** under the firewall option
   * Unselect the **Allow SSH traffic from** box if utilizing SSM Connect. If using ssh with keypair, limit the access to your IP address in the dropdown next to the checkbox.
   * Select the **Edit** option in the top right of the Network settings box. Rename the security group name (e.g,. `docker-builder-security-group`)
8. Under **Configure Storage**, up the capacity to **100GiB**
9. Drop down the **Advanced details** section
10. In the IAM instance profile dropdown, select the appropriate EC2 instance role with the required permissions.
11. Select the orange **Launch Instance** button on the right of the screen and wait for instance to launch.

### Connect to instance
**Using Session Manager**
1. Navigate to your list of instances and select the checkmark next to the `docker-builder` instance we just launched.
2. Click on the **Connect** button at the top of the screen. This will navigate to a new pane.
3. Select the **Session Manager** tab, then hit the orange **Connect** button in the bottom right.
4. Activate `bash` terminal and navigate to the home directory
```shell
bash
cd /home/ssm-user/
```

**Using ssh with keypairs**
1. In the **Details** pane for the instance, find and copy the **Public IPV4 DNS** value.
2. Open your terminal, connect to the instance, and navigate to the home directory:
```shell
ssh -i <key-pair> ec2-user@<ipv4-dns-address>
cd /home/ec2-user/
```

### Install software
Next, we need to install Docker, python, and git onto the instance.

```shell
sudo yum update
sudo yum install -y docker python git
```

We will install the code commit plugin for python in order to clone the repository

```shell
python -m ensurepip
python -m pip install --upgrade pip git-remote-codecommit
```

Clone the code commit repo. We will use the HTTPS(GRC) clone type. To get this URL:
1. Navigate to the CodeCommit console
2. Select the repository you want to clone.
3. In the top bar, select the **Clone URL** drop down and select the **Clone HTTPS(GRC)** option. This will copy the repo url into your clipboard.

```shell
git clone <https-grc-url>
cd <repo-name>
```

Finally, we need to start up the docker daemon and grant permissions for access.

```shell
sudo service docker start
sudo usermod -a -G docker <ssm-user/ec2-user>
sudo reboot
```

Add the user based on your connection type. `ssm-user` if using Session Manger access or `ec2-user` if accessing via ssh.
Once the instance is rebooted, reconnect to the instance and run a docker command to ensure you have access.

```shell
docker images
```
## Build the Docker Containers

1. If building locally, ensure account variables are set. If using EC2, ensure appropriate role is attached to the instance.
2. Navigate to the root of this repository in your terminal.
3. Login to the SageMaker ECR repo in order to pull them
   * Note, the below account is for the commercial regions (e.g., `us-west-2`). See [How to Update](#how-to-update) on how to identify accounts for other regions.
   * Details on pushing/pulling from ECR can be found in [this documentation](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html)
```shell
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com
```

4. Build the docker contains.
    * We support building either cpu or gpu optimized inference containers
```shell
docker build --tag training --platform linux/amd64 -f docker/PytorchTraining.dockerfile .
# To build CPU optimized inference container
docker build --tag inference --platform linux/amd64 --build-arg DEVICE=cpu -f docker/PytorchInference.dockerfile .
# To build GPU optimized inference container
docker build --tag inference --platform linux/amd64 --build-arg DEVICE=gpu -f docker/PytorchInference.dockerfile .
```

## Create ECR Repository
Next, we need to create a container repository for storing these containers in your account.

1. Navigate to the ECR Console page in your account
2. Click the orange **Create repository** button
3. Ensure the visibility is set to **Private**.
4. Name the repository:
    * Training: `pytorch-training`
    * Inference: `pytorch-inference`
* Enable **Scan on push** if required or as an added layer of security
* Click on **Create repository**

Repeat the steps above to create a training and inference repository.
We will use the fully qualified container name for pushing and pulling to ECR.
These look like:
```text
<account-id>.dkr.ecr.<region>.amazonaws.com/<repository-name>:<tag>
```

## Push the Docker Containers
Push commands for ECR can be reviewed in the Console by going to your ECR Repository and clicking the **View push commands** button in the top right of the repository information screen.
They can also be viewed more generally in the [documentation](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html).

1. Log docker into our account

```shell
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com
```
Replace `<region>` and `<aws_account_id>` with the values for your account.

2. Retag the built containers to match your ECR repository name.

```shell
# Retag the training container
docker tag training <aws_acount_id>.dkr.ecr.<region>.amazonaws.com/pytorch-training:2.2-gpu-py310
# Retag the inference container
docker tag inference <aws_acount_id>.dkr.ecr.<region>.amazonaws.com/pytorch-inference:2.2-<device>-py310
```

Update the tag to match the build traits as needed.

3. Push containers into ECR

```shell
# Push training container
docker push <aws_acount_id>.dkr.ecr.<region>.amazonaws.com/pytorch-training:2.2-gpu-py310
# Push inference container
docker push <aws_acount_id>.dkr.ecr.<region>.amazonaws.com/pytorch-inference:2.2-<device>-py310
```

## Clean Up

### Terminate EC2 Instance
If building on an EC2 instance, it is recommended to terminate the instance after testing the built containers.
You may optionally stop the instance instead.
You will no longer be paying the on-demand pricing for the instance type, but you still incur costs for the attached harddrive space.

1. Navigate to the EC2 Consoel
2. Click the box next to the `docker-builder` instance.
3. Click the **Instance State** drop down button and select **Stop Instance** or **Terminate Instance** as desired.

### Clean up local images
The training image, specifically, can be quite large and take up approximately 20GB of disk space on your development device.
Once testing is completed, we recommend deleting these images to recover some space.

1. Remove the built containers
```shell
docker image rm training inference
```

2. Rune and prune job to clear up dangling image resources pulled for building
```shell
docker image prune
```

Resources for docker commands found here:
* [Remove Images](https://docs.docker.com/reference/cli/docker/image/rm/)
* [Prune Images](https://docs.docker.com/config/pruning/)
* [Prune System](https://docs.docker.com/reference/cli/docker/system/prune/)

# Using the Containers
Once the containers are in ECR, using them is straight forward within SageMaker.
Typically, it means replacing the `framework_version` and `py_version` parameters in the SageMaker Framework modules with `image_uri`.

## Training
In the `LaunchTrainingJob.ipynb` and `TrainCountExperiment.ipynb` notebooks, we will update the estimator creation calls.
In each notebook, locate where the PyTorch estimator is created and modify it as seen below:

```python
estimator = PyTorch(
    entry_point="train.py",
    # framework_version="2.2",  # Comment/delete this line
    # py_version="py310",       # Comment/delete this line
    # Add this parameter and set is as the ECR repository name for the training container
    image_uri="<aws_acount_id>.dkr.ecr.<region>.amazonaws.com/pytorch-training:2.2-gpu-py310",
    ...  # Keep rest as normal
)
```

Training will now utilize your built container and proceed as normal. No further changes are required.

## Inference
We will need to repackage existing models to utilize this new container.
The container image to use for inference is defined by the model package as a whole.

We will update the `PackageModel.ipynb` notebook to utilize the new inference container.

In the **Package Model** section, update the first cell to include an additional imports.

```python
from sagemaker import get_execution_role, Session
from sagemaker.pytorch import PyTorch, PyTorchModel
```

In the final cell, we will create the model directly rather than via the estimator helper function.

```python
# Create the Model
training_estimator = PyTorch.attach(training_job_name)

# Comment out the original code for create the model_package
# model_package = training_estimator.create_model(
#     name=model_name,
#     entry_point=inference_entrypoint,
#     # source_dir=src_dir,
#     role=role,
#     env=env_variables,
#     model_server_workers=workers_per_model,
# )

# Make the model directly and reference the required fields from the estimator where needed
model_package = PyTorchModel(
    name=model_name,
    model_data=training_estimator.model_data,
    entry_point=inference_entrypoint,
    source_dir=(src_dir or training_estimator._model_source_dir()),
    role=role,
    env=env_variables,
    model_server_workers=workers_per_model,
    image_uri="638418947120.dkr.ecr.us-east-1.amazonaws.com/llnl-pytorch-inference:2.2-cpu-py310",
    sagemaker_session=Session(),
)

model_package.create(deploy_instance_type)
```

# How to Update

## Requirements Package
If updating one of the additional packages installed by the requirements file, re-run the container building steps and push the update.
Optionally, you can update `src/requirements.txt` and pin the specific version of the package you want installed.

## Update PyTorch Version
### Option 1) Use SageMaker container using the required PyTorch Version
1. Identify the SageMaker Container utilizing the needed version of PyTorch. Can use SageMaker's image_uri's package:
```python
from sagemaker import image_uris
image_uris.retrieve(framework="pytorch", region='<region>', version='<version>>', image_scope="training")
```
2. Update the docker file's `FROM` line to use this ECR container
```dockerfile
FROM <ecr-image-name-here>
```
3. Re-build the container with updated dockerfile

### Option 2) Build full container using SageMaker dockerfile as template
If SageMaker does not currently provide a container with the required version, you can build the container yourself.
SageMaker dockerfiles are hosted in Github in the [aws/deep-learning-containers](https://github.com/aws/deep-learning-containers/tree/master/pytorch) repository.
Identify the docker file with the closest version match as your template, and copy the contents.
These will be under the `docker` folder under `training` for `inference` respectively.
For example, the docker file for PyTorch 2.3 training is found [here](https://github.com/aws/deep-learning-containers/blob/master/pytorch/training/docker/2.3/py3/cu121/Dockerfile.gpu)

Once updated, re-build the container using the same steps as above.

## Update other requirement such as Python version or OS Version
Follow the steps in the [Option 2 for updating pytorch](#option-2-build-full-container-using-sagemaker-dockerfile-as-template).
Update the necessary version, and potentially the base nvidia container for updated linux versions.
