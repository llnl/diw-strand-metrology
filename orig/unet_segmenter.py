import argparse
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.multiprocessing as mp
import torch.distributed as dist




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
#print('num_gpus = ', num_gpus)

import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, input_channels):
        super(UNet, self).__init__()

        # Contracting Path
        self.enc1 = self.contract_block(input_channels, 64)
        self.enc2 = self.contract_block(64, 128)
        
        # Expansive Path
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.expand_block(128, 128, 64)

        
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.expand_block(32 + input_channels, 64, 32)


        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def contract_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def expand_block(self, in_channels, middle_channels, out_channels):  # Add middle_channels argument
        block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),  # Use middle_channels here
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),  # And here
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        
        upconv1 = self.upconv1(enc2)

        #print("upconv1 shape:", upconv1.shape)
        #print("enc1 shape:", enc1.shape)
        #combined_tensor = torch.cat([upconv1, enc1], 1)
        #print("Combined tensor shape:", combined_tensor.shape)

        dec1 = self.dec1(torch.cat([upconv1, enc1], 1))
        
        upconv2 = self.upconv2(dec1)
        dec2 = self.dec2(torch.cat([upconv2, x], 1))  # connecting back to original input

        return torch.sigmoid(self.final_conv(dec2))

class FCN(nn.Module):
    def __init__(self, input_channels):
        super(FCN, self).__init__()
        
        # Encoding layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoding layers
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        )
        
        self.dec3 = nn.Conv2d(64, 1, kernel_size=1)  # Output layer
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.dec1(x2)
        x4 = self.dec2(x3)
        x5 = self.dec3(x4)
        return torch.sigmoid(x5)



class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # If input has multiple channels, calculate the loss for each channel separately
        if inputs.size(1) > 1:
            total_loss = 0.0
            for channel in range(inputs.size(1)):
                input_channel = inputs[:, channel, :, :]
                target_channel = targets[:, 0, :, :]  # Assuming target has a single channel
                total_loss += self.loss(input_channel, target_channel)
            return total_loss / inputs.size(1)
        else:
            return self.loss(inputs, targets)


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, image_mode='RGB'):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_mode = image_mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_mode = 'RGB' if self.image_mode == 'RGB' else 'L'
        #print(f"image_mode = {img_mode}, {self.image_paths[idx]}")
        img = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx]) #.convert('L')
        #print("before: ", np.unique(np.array(mask)), np.array(mask).dtype)
        #print("loc 1: ", np.array(img).dtype, np.max(np.array(img)), np.min(np.array(img)), np.mean (np.array(img))) 

        # Print the filenames here
        #img_name = os.path.basename(self.image_paths[idx])
        #mask_name = os.path.basename(self.mask_paths[idx])
        #print(f"Fetching Image: {img_name}, Mask: {mask_name}")
        
        # Convert directly to tensor, np.array converts to int32 and then torch converts to float32
        img_tensor = transforms.ToTensor()(img) #torch.from_numpy(np.array(img)) #transforms.ToTensor()(img) #careful toTensor normalizes by 255
        mask_tensor = transforms.ToTensor()(mask)*255. #torch.from_numpy(np.array(mask)) #
        #print("loc2 mask: ", np.array(mask_tensor).dtype, np.unique(mask_tensor))
        #print("loc 2 img: ", np.array(img_tensor).dtype, np.max(np.array(img_tensor)), np.min(np.array(img_tensor)), np.mean (np.array(img_tensor)))

        # Add channel dimension
        #img_tensor = img_tensor.unsqueeze(1)  # Add channel dimension, resulting in [B, C, H, W]
        #print(img_tensor.shape, mask_tensor.shape)

       # Ensure tensors are of type float32 (though they should be by default)
        img_tensor = img_tensor.float()
        mask_tensor = mask_tensor.float()

        # Normalize the image tensor by its max value to ensure the range is [0, 1]
        img_tensor /= img_tensor.max()
        mask_tensor /= mask_tensor.max()

        #print("loc 3 mask: ", np.array(mask_tensor).dtype, np.unique(mask_tensor))
        #print("loc 3 img: ", np.array(img_tensor).dtype, np.max(np.array(img_tensor)), np.min(np.array(img_tensor)), np.mean (np.array(img_tensor)))

        # Convert mask values: 112/255 to 1 and 1 to 0
        mask_tensor[mask_tensor == 1] = 2  # Set values equal to 1 (i.e., 255/255) to an intermediate value, 2
        mask_tensor[mask_tensor < 0.5] = 1  # Convert values less than 0.5 (i.e., 112/255) to 1
        mask_tensor[mask_tensor == 2] = 0  # Convert the intermediate value, 2, to 0

        #print("after last: ", mask_tensor.min(), mask_tensor.max())
        
        # Apply transforms if available
#        if self.transform:
#            img_tensor = self.transform(img_tensor)
#            mask_tensor = self.transform(mask_tensor)
        return img_tensor, mask_tensor

def get_data_loaders(args, image_paths, mask_paths, batch_size, image_mode='RGB'):
    # Split data into train, validation, and test sets
    img_train, img_temp, mask_train, mask_temp = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
    img_val, img_test, mask_val, mask_test = train_test_split(img_temp, mask_temp, test_size=0.5, random_state=42)

    # Define different sets of transformations for RGB and grayscale images
    transform = transforms.Compose([
#            transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = SegmentationDataset(img_train, mask_train, transform=transform, image_mode=image_mode)
    val_dataset = SegmentationDataset(img_val, mask_val, transform=transform, image_mode=image_mode)
    test_dataset = SegmentationDataset(img_test, mask_test, transform=transform, image_mode=image_mode)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)


    return train_loader, val_loader, test_loader

def he_initialization(layer):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("NCCL", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def main(rank, args):
    setup(rank, torch.cuda.device_count())
    
    # Get list of paths in the folders
    image_paths = sorted([os.path.join(args.image_folder, fname) for fname in os.listdir(args.image_folder) if fname.endswith(('jpeg', 'png', 'jpg'))])
    mask_paths = sorted([os.path.join(args.mask_folder, fname) for fname in os.listdir(args.mask_folder) if fname.endswith(('jpeg', 'png', 'jpg'))])


    # Determine the number of input channels based on the image mode
    input_channels = 3 if args.image_mode == 'RGB' else 1

  # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        args, image_paths, mask_paths, args.batch_size, args.image_mode)  # Define train_loader here

    # Initialize the model
    model = UNet(input_channels=input_channels)
    model.apply(he_initialization)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Usage:
    print(f"The model has {count_parameters(model):,} trainable parameters.")


    # If multiple GPUs are available, wrap the model with DataParallel
    if num_gpus > 1:
        model = nn.DataParallel(model)
    
    # Move the model to the device
    model.to(device)

    # Create lists to store training and validation loss/accuracy for plotting
    train_losses = []
    val_losses = []

    # Define an optimizer and a loss function
    if not args.use_zero:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        # Only supported when using DDP
        # See https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=args.learning_rate,
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Prevent gradient underflow when computing loss in half precision
    grad_scaler = GradScaler(enabled=args.use_amp)

    # Using BCEWithLogitsLoss as the loss function
    criterion = BCEWithLogitsLoss()

    criterion.to(device)  # Move to device

    # Train the model with tqdm progress bar
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)  # Get the total number of batches

        # Initialize a batch counter before the loop
        batch_counter = 0 

        print(f"Starting Epoch {epoch + 1}")  # Print epoch start message

        # Initialize tqdm progress bar
        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}", ncols=100, position=0, leave=True) as pbar:
            for batch in train_loader:
                images, masks = batch[0].to(device), batch[1].to(device)
                #print(images.shape, masks.shape)  # Should be [batch_size, C, H, W]
                #images, masks = images.double(), masks.double()  # Convert to double precision

                # Print data type and min/max values
                #print("Image Data Type:", images.dtype)
                #print("Image Min Value:", images.min())
                #print("Image Max Value:", images.max())

                #print("Mask Data Type:", masks.dtype)
                #print("Mask Min Value:", masks.min())
                #print("Mask Max Value:", masks.max())
                # See https://pytorch.org/docs/stable/notes/amp_examples.html
                with autocast(enabled=args.use_amp):
                    outputs = model(images)

                    # In the training loop, after calculating the outputs:
                    #print("Model Outputs Min Value:", outputs.min().item())
                    #print("Model Outputs Max Value:", outputs.max().item())
                    #print("Masks Min Value:", masks.min().item())
                    #print("Masks Max Value:", masks.max().item())
                    # Increment the batch counter
                    batch_counter += 1

                    # Print the information every N batches
                    if batch_counter % 10 == 0:
                        print(f"\nModel Outputs - Min: {outputs.min().item()}, Max: {outputs.max().item()}, Mean: {outputs.mean().item()}")
                        print(f"Masks - Min: {masks.min().item()}, Max: {masks.max().item()}, Mean: {masks.mean().item()}")

                    loss = criterion(outputs, masks)

                # After calculating the loss for a batch:
                print("Batch Loss:", loss.item())

                optimizer.zero_grad()

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("NaN or Inf found in loss.")
                    continue

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("NaN or Inf found in model outputs.")
                    continue

                if torch.isnan(masks).any() or torch.isinf(masks).any():
                    print("NaN or Inf found in masks.")
                    continue

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                grad_scaler.scale(loss).backward()

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                grad_scaler.step(optimizer)

                # Updates the scale for next iteration.
                grad_scaler.update()

                train_loss += loss.item()

                pbar.update(1)  # Update the progress bar by 1 step (1 batch processed)

        # Calculate the average training loss for the epoch
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)  # Add this line to store the training loss

        # Validation (you might want to add validation and other metrics)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch
                images, masks = images.to(device), masks.to(device)  # Move to device
                #images, masks = images.to(device).double(), masks.to(device).double()

                outputs = model(images)

                #print("Image batch shape:", images.shape)
                #print("Mask batch shape:", masks.shape)

                loss = criterion(outputs, masks)

                val_loss += loss.item()

        # Calculate the average validation loss for the epoch
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)  # Add this line to store the validation loss
        scheduler.step(avg_val_loss)  # Update learning rate based on validation loss


        # Print loss for the current epoch
        print(f"Epoch {epoch + 1} completed. Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")


        # Save model weights every epoch
        checkpoint_path = os.path.join(args.model_dir, 'model_epoch.pt')
        
        # Check if the model is an instance of DataParallel
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)

    # Generate training and validation loss plots
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    output_file_path = os.path.join(args.output_data_dir, 'loss_plot.png')
    plt.savefig(output_file_path, dpi=300)

    # After the training and validation phases are complete
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch
            images, masks = images.to(device), masks.to(device)  # Move to device
            #images, masks = images.to(device).double(), masks.to(device).double()

            outputs = model(images)

            loss = criterion(outputs, masks)
            test_loss += loss.item()

    # Calculate the average test loss for the evaluation
    avg_test_loss = test_loss / len(test_loader)
    print(f'Average Test Loss: {avg_test_loss}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script for U-Net segmentation model')

    parser.add_argument('--image_folder', type=str, default=os.environ['SM_CHANNEL_TRAIN_IMAGE'], help='Path to the folder containing images')
    parser.add_argument('--mask_folder', type=str, default=os.environ['SM_CHANNEL_TRAIN_MASK'], help='Path to the folder containing masks')
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'], help='Location for saving output artifacts')
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'], help='Location for saving the model')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--image_mode', type=str, choices=['RGB', 'grayscale'], default='grayscale', help='Image mode for reading images (RGB or grayscale)')
    parser.add_argument('--use_zero', type=bool, default=False, help='Enable ZeRO optimizer')
    parser.add_argument('--use_amp', type=bool, default=False, help='Enable Mixed Precision')

    args = parser.parse_args()
    
    # Modify the UNet input_channels based on the selected image_mode
    if args.image_mode == 'grayscale':
        args.input_channels = 1
    else:  # If it's not grayscale, it will be RGB.
        args.input_channels = 3
        
    world_size = torch.cuda.device_count()
    
    mp.spawn(main,
             args=(args, ),
             nprocs=world_size,
             join=True
            )
