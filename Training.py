import os
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# -------------------------
# 1. Configuration
# -------------------------

EXCEL_PATH = r"C:\Users\Guanli\OneDrive - National University of Singapore\BPS5231\Final_CNN\augmented_dataset.xlsx"
IMG_DIR    = r"C:\Users\Guanli\OneDrive - National University of Singapore\BPS5231\Final_CNN\plan"

BATCH_SIZE = 16
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
RANDOM_STATE = 42
IMG_SIZE = 256  # resize from 512x512 to 256x256 for efficiency


# -------------------------
# 2. Dataset definition
# -------------------------

class LightingDatasetNoID(Dataset):
    """
    Each sample consists of:
        Inputs:
            - Plan image (RGB, resized)
            - Tabular: normalized occupant_count, seat_00..seat_23 (24 seats)
              (NO id in the inputs)
        Outputs:
            - Normalized task_00..task_23
            - Normalized ceiling_1..ceiling_3
            - Normalized Power
    """
    def __init__(self, df, img_dir, indices,
                 task_cols, ceiling_cols, seat_cols,
                 max_occupant, max_power):
        self.df = df.reset_index(drop=True)
        self.indices = indices
        self.img_dir = img_dir

        self.task_cols = task_cols
        self.ceiling_cols = ceiling_cols
        self.seat_cols = seat_cols

        self.max_occupant = float(max_occupant)
        self.max_power = float(max_power)

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])  # -> [-1,1]
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row_idx = self.indices[idx]
        row = self.df.iloc[row_idx]

        # ---- Image (we still use id ONLY to locate file) ----
        img_id = int(row["id"])
        img_filename = f"{img_id}.bmp"
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # ---- Tabular input (NO id here) ----
        occ_norm = row["occupant_count"] / self.max_occupant if self.max_occupant > 0 else 0.0
        seat_values = row[self.seat_cols].values.astype(np.float32)  # 0 or 1

        # tabular = [occupant_count_norm, seat_00..seat_23]  -> length = 1 + 24 = 25
        tabular = np.concatenate([[occ_norm], seat_values], axis=0).astype(np.float32)
        tabular = torch.tensor(tabular, dtype=torch.float32)

        # ---- Targets (normalized) ----
        # Lighting levels 0..5 -> divide by 5
        task_vals = row[self.task_cols].values.astype(np.float32) / 5.0
        ceiling_vals = row[self.ceiling_cols].values.astype(np.float32) / 5.0

        # Power normalized by max_power
        power_val = row["Power"] / self.max_power if self.max_power > 0 else 0.0

        targets = np.concatenate([task_vals, ceiling_vals, [power_val]], axis=0)
        targets = torch.tensor(targets, dtype=torch.float32)

        return image, tabular, targets


# -------------------------
# 3. Model definition (image + occupant_count + seats)
# -------------------------

class LightingCNNNoID(nn.Module):
    def __init__(self, tabular_input_dim=25, output_dim=28):
        """
        tabular_input_dim:
            1 (occupant_count_norm) + 24 seats = 25
        output_dim:
            24 tasks + 3 ceilings + 1 power = 28
        """
        super(LightingCNNNoID, self).__init__()

        # Image branch: small CNN
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

        # Global pooling
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
        # Image branch
        x_img = self.img_conv(image)           # [B, 64, 32, 32]
        x_img = self.img_pool(x_img)           # [B, 64, 4, 4]
        x_img = torch.flatten(x_img, 1)        # [B, 1024]

        # Tabular branch
        x_tab = self.tabular_net(tabular)      # [B, 64]

        # Fusion
        x = torch.cat([x_img, x_tab], dim=1)   # [B, 1088]
        out = self.fusion_net(x)               # [B, 28]
        return out


# -------------------------
# 4. Training loop
# -------------------------

def train_model_noid():
    # ----- Load data -----
    df = pd.read_excel(EXCEL_PATH)

    # Column definitions
    task_cols = [f"task_{i:02d}" for i in range(24)]  # task_00 ... task_23
    seat_cols = [f"seat_{i:02d}" for i in range(24)]  # seat_00 ... seat_23
    ceiling_cols = ["ceiling_1", "ceiling_2", "ceiling_3"]

    # Basic checks
    for col in ["id", "occupant_count", "Power"]:
        if col not in df.columns:
            raise ValueError(f"Missing column in Excel: {col}")
    for col in task_cols + seat_cols + ceiling_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in Excel: {col}")

    max_occupant = df["occupant_count"].max()
    max_power = df["Power"].max()

    # Train/validation split based on indices
    indices = np.arange(len(df))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=VAL_SPLIT,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Print which ids are used for validation (testing)
    print("Number of validation (testing) samples:", len(val_indices))
    val_ids = df.iloc[val_indices]["id"].tolist()
    print("Validation sample IDs:", val_ids)

    train_dataset = LightingDatasetNoID(
        df, IMG_DIR, train_indices,
        task_cols, ceiling_cols, seat_cols,
        max_occupant=max_occupant,
        max_power=max_power
    )

    val_dataset = LightingDatasetNoID(
        df, IMG_DIR, val_indices,
        task_cols, ceiling_cols, seat_cols,
        max_occupant=max_occupant,
        max_power=max_power
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ----- Model, loss, optimizer -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tabular_input_dim = 1 + len(seat_cols)  # occ_norm + seat_00..seat_23
    output_dim = len(task_cols) + len(ceiling_cols) + 1  # 24 + 3 + 1 = 28

    model = LightingCNNNoID(tabular_input_dim=tabular_input_dim, output_dim=output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # ----- Training loop -----
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for images, tabular, targets in train_loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images, tabular)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, tabular, targets in val_loader:
                images = images.to(device)
                tabular = tabular.to(device)
                targets = targets.to(device)

                outputs = model(images, tabular)
                loss = criterion(outputs, targets)

                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        print(f"Epoch [{epoch:03d}/{NUM_EPOCHS}]  "
              f"Train Loss: {avg_train_loss:.6f}  |  Val Loss: {avg_val_loss:.6f}")

    # ----- Save model and scaling factors -----
    save_dict = {
        "model_state_dict": model.state_dict(),
        "max_occupant": max_occupant,
        "max_power": max_power,
        "task_cols": task_cols,
        "ceiling_cols": ceiling_cols,
        "seat_cols": seat_cols,
        "use_id_in_input": False
    }
    torch.save(save_dict, "lighting_cnn_model_noid.pth")
    print("Model saved to lighting_cnn_model_noid.pth")


if __name__ == "__main__":
    train_model_noid()
