import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np


# ============================================================
#                 USER INPUTS (EDIT HERE)
# ============================================================

# 1) Path to the trained model (from your retraining)
MODEL_PATH = r"lighting_cnn_model_noid.pth"

# 2) Path to the BMP plan you want to use as input
IMG_PATH = r"C:\Users\Guanli\OneDrive - National University of Singapore\BPS5231\Final_CNN\plan\64.bmp"

# 3) Overall occupancy (how many people are in the room)
OCCUPANT_COUNT = 10  # example

# 4) Seat occupancy: list of 24 values (0 or 1) for seat_00 ... seat_23
#    Make sure the length is 24; you can adjust this pattern as needed
SEAT_OCCUPANCY = [
    0, 0, 0, 0,  # seat_00..seat_03
    0, 0, 1, 1,  # seat_04..seat_07
    1, 1, 0, 1,  # seat_08..seat_11
    0, 0, 1, 1,  # seat_12..seat_15
    0, 1, 0, 0,  # seat_16..seat_19
    0, 1, 1, 0   # seat_20..seat_23
]

# (Optional sanity check: you can ensure sum(SEAT_OCCUPANCY) == OCCUPANT_COUNT)


# ============================================================
#                 MODEL / INFERENCE CODE
# ============================================================

IMG_SIZE = 256  # must match training


class LightingCNNNoID(nn.Module):
    """
    Same architecture as used in training:
    - Image CNN branch
    - Tabular branch: [occupant_count_norm, seat_00..seat_23]  -> length 25
    - Output: 24 tasks + 3 ceilings + 1 power = 28
    """
    def __init__(self, tabular_input_dim=25, output_dim=28):
        super(LightingCNNNoID, self).__init__()

        # Image branch
        self.img_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256 -> 128

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
        )

        self.img_pool = nn.AdaptiveAvgPool2d((4, 4))  # 64x32x32 -> 64x4x4
        img_feature_dim = 64 * 4 * 4  # 1024

        # Tabular branch
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # Fusion MLP
        fusion_input_dim = img_feature_dim + 64
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, output_dim)
        )

    def forward(self, image, tabular):
        x_img = self.img_conv(image)
        x_img = self.img_pool(x_img)
        x_img = torch.flatten(x_img, 1)

        x_tab = self.tabular_net(tabular)

        x = torch.cat([x_img, x_tab], dim=1)
        out = self.fusion_net(x)
        return out


def load_model_and_metadata():
    """
    Load model weights and scaling factors from checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # IMPORTANT: allow full checkpoint (not weights-only)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    task_cols = checkpoint["task_cols"]        # ['task_00', ..., 'task_23']
    ceiling_cols = checkpoint["ceiling_cols"]  # ['ceiling_1', 'ceiling_2', 'ceiling_3']
    seat_cols = checkpoint["seat_cols"]        # ['seat_00', ..., 'seat_23']

    max_occupant = checkpoint["max_occupant"]
    max_power = checkpoint["max_power"]

    tabular_input_dim = 1 + len(seat_cols)  # 1 (occ_norm) + 24 seats = 25
    output_dim = len(task_cols) + len(ceiling_cols) + 1  # 24 + 3 + 1 = 28

    model = LightingCNNNoID(tabular_input_dim=tabular_input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    return model, device, transform, task_cols, ceiling_cols, max_occupant, max_power


def predict_for_input(
    img_path,
    occupant_count,
    seat_occupancy,
    model,
    device,
    transform,
    task_cols,
    ceiling_cols,
    max_occupant,
    max_power
):
    """
    Inference using:
        - bmp image
        - occupant_count
        - seat_occupancy (list of 24 ints: 0 or 1)
    """
    # ---- Check seat occupancy length ----
    seat_array = np.asarray(seat_occupancy, dtype=np.float32)
    if seat_array.shape[0] != 24:
        raise ValueError(f"SEAT_OCCUPANCY must have length 24, got {seat_array.shape[0]}")

    # ---- Image ----
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # [1, 3, H, W]

    # ---- Tabular input ----
    occ_norm = occupant_count / max_occupant if max_occupant > 0 else 0.0

    tabular = np.concatenate([[occ_norm], seat_array], axis=0).astype(np.float32)
    tabular = torch.tensor(tabular, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 25]

    # ---- Inference ----
    with torch.no_grad():
        outputs = model(image, tabular)  # [1, 28]

    outputs = outputs.squeeze(0).cpu().numpy()

    n_tasks = len(task_cols)
    n_ceilings = len(ceiling_cols)

    tasks_norm = outputs[:n_tasks]                         # 0..1
    ceilings_norm = outputs[n_tasks:n_tasks + n_ceilings]  # 0..1
    power_norm = outputs[-1]                               # scaled by max_power

    # ---- Denormalize ----
    tasks = np.clip(tasks_norm * 5.0, 0.0, 5.0)
    ceilings = np.clip(ceilings_norm * 5.0, 0.0, 5.0)
    power = power_norm * max_power

    # Round light levels to nearest integer
    tasks_rounded = np.rint(tasks).astype(int)
    ceilings_rounded = np.rint(ceilings).astype(int)

    task_dict = {col: val for col, val in zip(task_cols, tasks_rounded)}
    ceiling_dict = {col: val for col, val in zip(ceiling_cols, ceilings_rounded)}

    result = {
        "occupant_count": occupant_count,
        "tasks_raw": tasks,
        "ceilings_raw": ceilings,
        "power_raw": float(power),
        "tasks_rounded": task_dict,
        "ceilings_rounded": ceiling_dict,
        "power": float(power)
    }
    return result


def main():
    # Load model and metadata
    (model, device, transform,
     task_cols, ceiling_cols,
     max_occupant, max_power) = load_model_and_metadata()

    # Run prediction for the user-specified inputs at the top
    result = predict_for_input(
        img_path=IMG_PATH,
        occupant_count=OCCUPANT_COUNT,
        seat_occupancy=SEAT_OCCUPANCY,
        model=model,
        device=device,
        transform=transform,
        task_cols=task_cols,
        ceiling_cols=ceiling_cols,
        max_occupant=max_occupant,
        max_power=max_power
    )

    print(f"\nPrediction for this input (occupant_count = {result['occupant_count']}):")

    print("\nPredicted desk lighting levels (rounded to 0–5):")
    for k, v in result["tasks_rounded"].items():
        print(f"  {k}: {v}")

    print("\nPredicted ceiling lighting levels (rounded to 0–5):")
    for k, v in result["ceilings_rounded"].items():
        print(f"  {k}: {v}")

    print(f"\nPredicted power: {result['power']:.2f} (same units as your Power column)\n")


if __name__ == "__main__":
    main()
