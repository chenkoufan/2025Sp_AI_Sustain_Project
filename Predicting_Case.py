import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

# ==============================
# 1. 路径配置（按需修改这里）
# ==============================

MODEL_PATH = r"32_300/32_300.pth"                  # 训练好的模型
CSV_PATH   = r"simulated_seats.csv"       # 你的占用情况 CSV
IMG_DIR    = r"test/plan"                 # bmp 平面图所在文件夹
OUTPUT_CSV = r"simulated_seats_with_control.csv"  # 输出文件名


IMG_SIZE = 256  # 必须与训练一致


# ==============================
# 2. 模型定义（与训练脚本一致）
# ==============================

class LightingCNNNoID(nn.Module):
    """
    与训练时相同的网络结构：
    - 图像分支 CNN
    - Tabular 分支: [occupant_count_norm, seat_00..seat_23]
    - 输出: 24 tasks + 3 ceilings + 1 power = 28
    """
    def __init__(self, tabular_input_dim=25, output_dim=28):
        super(LightingCNNNoID, self).__init__()

        # 图像分支
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

        # 表格分支
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 融合 MLP
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
        x_img = torch.flatten(x_img, 1)   # [B, 1024]

        x_tab = self.tabular_net(tabular) # [B, 64]

        x = torch.cat([x_img, x_tab], dim=1)  # [B, 1088]
        out = self.fusion_net(x)              # [B, 28]
        return out


# ==============================
# 3. 加载模型与元数据
# ==============================

def load_model_and_metadata():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    task_cols    = checkpoint["task_cols"]        # 训练时的 task 列名
    ceiling_cols = checkpoint["ceiling_cols"]     # ceiling 列名
    seat_cols    = checkpoint["seat_cols"]        # seat_00..seat_23
    max_occupant = checkpoint["max_occupant"]
    max_power    = checkpoint["max_power"]

    tabular_input_dim = 1 + len(seat_cols)
    output_dim        = len(task_cols) + len(ceiling_cols) + 1

    model = LightingCNNNoID(tabular_input_dim=tabular_input_dim,
                            output_dim=output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    return model, device, transform, task_cols, ceiling_cols, seat_cols, max_occupant, max_power


# ==============================
# 4. 单行数据的预测函数
# ==============================

def predict_for_row(row,
                    model,
                    device,
                    transform,
                    task_cols,
                    ceiling_cols,
                    seat_cols,
                    max_occupant,
                    max_power):
    """
    给定一行 df（含 id、occupant_count 和 seat_* 列），
    读取对应平面图 + 构造 tabular + 前向推理，
    返回：tasks_rounded (24,), ceilings_rounded (3,)
    """

    # ---- 读取 img ----
    img_id = int(row["id"])
    img_path = os.path.join(IMG_DIR, f"{img_id}.bmp")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found for id={img_id}: {img_path}")

    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # [1, 3, H, W]

    # ---- 构造 tabular 输入 ----
    occ = float(row["occupant_count"])
    # 简单 clamp，防止超过训练最大值
    if max_occupant > 0:
        if occ > max_occupant:
            print(f"Warning: occupant_count {occ} > training max {max_occupant}, clamped.")
        occ_norm = min(occ, max_occupant) / max_occupant
    else:
        occ_norm = 0.0

    # seat_* 列按训练保存的 seat_cols 顺序取
    seat_vals = row[seat_cols].values.astype(np.float32)
    tabular_np = np.concatenate([[occ_norm], seat_vals], axis=0).astype(np.float32)
    tabular = torch.tensor(tabular_np, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 25]

    # ---- 推理 ----
    with torch.no_grad():
        outputs = model(image, tabular)  # [1, 28]
    outputs = outputs.squeeze(0).cpu().numpy()

    n_tasks    = len(task_cols)
    n_ceilings = len(ceiling_cols)

    tasks_norm    = outputs[:n_tasks]
    ceilings_norm = outputs[n_tasks:n_tasks + n_ceilings]
    # power_norm = outputs[-1]   # 如果你暂时不需要 power，可以先不用

    # 反归一化 & 四舍五入到 0..5
    tasks    = np.clip(tasks_norm    * 5.0, 0.0, 5.0)
    ceilings = np.clip(ceilings_norm * 5.0, 0.0, 5.0)

    tasks_rounded    = np.rint(tasks).astype(int)
    ceilings_rounded = np.rint(ceilings).astype(int)

    return tasks_rounded, ceilings_rounded


# ==============================
# 5. 主过程：遍历 CSV，写回控制策略
# ==============================

def main():
    # 1) 读入 simulated_seats.csv
    df = pd.read_csv(CSV_PATH)

    # 2) 加载模型和元数据
    (model, device, transform,
     task_cols, ceiling_cols, seat_cols,
     max_occupant, max_power) = load_model_and_metadata()

    # 3) 先创建要写入的列（根据你想要的列名）
    #    这里按你的描述用: task_light0..task_light23, ceiling_light1..ceiling_light3
    task_light_cols    = [f"task_light{i}"    for i in range(24)]
    ceiling_light_cols = [f"ceiling_light{j}" for j in range(1, 4)]

    for col in task_light_cols + ceiling_light_cols:
        if col not in df.columns:
            df[col] = np.nan   # 先占位

    # 4) 遍历每一行，做预测并填入
    for idx, row in df.iterrows():
        try:
            tasks_rounded, ceilings_rounded = predict_for_row(
                row,
                model,
                device,
                transform,
                task_cols,
                ceiling_cols,
                seat_cols,
                max_occupant,
                max_power
            )
        except FileNotFoundError as e:
            print(e)
            continue

        # 写回这行的 task_light*
        for i in range(24):
            df.at[idx, task_light_cols[i]] = int(tasks_rounded[i])

        # 写回这行的 ceiling_light*
        for j in range(3):
            df.at[idx, ceiling_light_cols[j]] = int(ceilings_rounded[j])

        # ===== NEW: 计算 optimized_power =====
        task_sum = np.sum(tasks_rounded)
        ceiling_sum = np.sum(ceilings_rounded)

        optimized_power = task_sum * 1.5 + ceiling_sum * 4.4 * 3
        df.at[idx, "optimized_power"] = optimized_power

        # ===== 2 计算 previous_power（基线策略） =====
        # 从这一行读取座位占用情况（0/1），按 seat_cols 顺序
        seat_vals = row[seat_cols].values.astype(float)

        # occupied_seats = 有人的座位数量
        occupied_seats = int(np.sum(seat_vals))

        # 区域划分：0-7, 8-15, 16-23
        region1_has_occ = np.sum(seat_vals[0:8])   > 0  # seat_00..seat_07
        region2_has_occ = np.sum(seat_vals[8:16])  > 0  # seat_08..seat_15
        region3_has_occ = np.sum(seat_vals[16:24]) > 0  # seat_16..seat_23

        ceiling_on_count = (
            int(region1_has_occ) +
            int(region2_has_occ) +
            int(region3_has_occ)
        )

        # previous_power = 每个有人座位4.5W + 每个亮着的ceiling 66W
        previous_power = occupied_seats * 4.5 + ceiling_on_count * 66.0
        df.at[idx, "previous_power"] = float(previous_power)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1} rows...")

    # 5) 保存新的 CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDone. Saved with control strategy to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
