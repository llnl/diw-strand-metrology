from torch import nn


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
                target_channel = targets[
                    :, 0, :, :
                ]  # Assuming target has a single channel
                total_loss += self.loss(input_channel, target_channel)
            return total_loss / inputs.size(1)
        else:
            return self.loss(inputs, targets)
