from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
      
        return F.cross_entropy(logits, target)

class DetectionLoss(nn.Module):
    def __init__(self, seg_weight=1.0, depth_weight=1.0, reduction='mean'):
        super().__init__()
        self.seg_weight = seg_weight
        self.depth_weight = depth_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, logits, targets, pred_depth, true_depth):
        """
        Args:
            logits: (b, num_classes, h, w) - raw segmentation logits
            targets: (b, h, w) - ground truth segmentation labels
            pred_depth: (b, h, w) - predicted depth
            true_depth: (b, h, w) - ground truth depth
        Returns:
            combined loss value
        """
        seg_loss = self.ce_loss(logits, targets)
        depth_loss = self.mse_loss(pred_depth, true_depth)
        
        return self.seg_weight * seg_loss + self.depth_weight * depth_loss

class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        z = self.conv_layers(z)
        z = z.view(z.size(0), -1)
        logits = self.fc(z)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)

class ResidualBlock(nn.Module):
    """Basic Residual Block with two Conv layers."""
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip Connection (Downsampling if needed)
        self.skip = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        return self.relu(out)

class Detector2(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.enc1 = ResidualBlock(in_channels, 16, downsample=True)  # (b, 16, h/2, w/2) 48
        self.enc2 = ResidualBlock(16, 32, downsample=True)  # (b, 32, h/4, w/4) 24
        self.enc3 = ResidualBlock(32, 64, downsample=True)  # (b, 64, h/8, w/8) 12
        self.enc4 = ResidualBlock(64, 128, downsample=True)  # (b, 128, h/16, w/16) 6

        # Bottleneck
        self.bottleneck = ResidualBlock(128, 256)  # (b, 256, h/16, w/16) 6

        # Decoder
        self.dec4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # (b, 128, h/8, w/8) 12
        self.up4 = ResidualBlock(256, 128)  # (b, 128, h/8, w/8) 12

        self.dec3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # (b, 64, h/4, w/4) 24
        self.up3 = ResidualBlock(128, 64)  # (b, 64, h/4, w/4)

        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # (b, 32, h/2, w/2)
        self.up2 = ResidualBlock(64, 32)  # (b, 32, h/2, w/2)

        self.dec1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # (b, 16, h, w)
        self.up1 = ResidualBlock(32, 16)  # (b, 16, h, w)

        # Segmentation Head (Final Output for Segmentation)
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)

        # Depth Head (Final Output for Depth Prediction)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        """Forward pass: Returns segmentation logits and depth prediction."""
        x = (x - self.input_mean) / self.input_std  # Normalize input

        # Encoding
        enc1 = self.enc1(x)  # (b, 16, h/2, w/2)
        enc2 = self.enc2(enc1)  # (b, 32, h/4, w/4)
        enc3 = self.enc3(enc2)  # (b, 64, h/8, w/8)
        enc4 = self.enc4(enc3)  # (b, 128, h/16, w/16)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # (b, 256, h/16, w/16)

        # Decoding with skip connections
        dec4 = self.dec4(bottleneck)  # (b, 128, h/8, w/8)
        up4 = self.up4(torch.cat([dec4, enc4], dim=1))  # (b, 128, h/8, w/8)

        dec3 = self.dec3(up4)  # (b, 64, h/4, w/4)
        up3 = self.up3(torch.cat([dec3, enc3], dim=1))  # (b, 64, h/4, w/4)

        dec2 = self.dec2(up3)  # (b, 32, h/2, w/2)
        up2 = self.up2(torch.cat([dec2, enc2], dim=1))  # (b, 32, h/2, w/2)

        dec1 = self.dec1(up2)  # (b, 16, h, w)
        up1 = self.up1(torch.cat([dec1, enc1], dim=1))  # (b, 16, h, w)

        # Final Outputs
        logits = self.segmentation_head(up1)  # (b, num_classes, h, w)
        raw_depth = self.depth_head(up1).squeeze(1)  # (b, h, w)

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth

class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)  # Ensure final (b, 1, h, w)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        enc = self.encoder(z)
        logits = self.decoder(enc)
        raw_depth = self.depth_head(enc).squeeze(1)
        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
    "detector2" : Detector2
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
