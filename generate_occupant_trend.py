import pandas as pd
import numpy as np

# ============================================================
#               用户需要修改的参数
# ============================================================
INPUT_CSV = "robod-master/Data/combined_Room4 - only count.csv"      # 输入：记录 timestamp 和 occupant_count 的文件
OUTPUT_CSV = "simulated_seats.csv"    # 输出：生成的座位分布文件
N_SEATS = 24                          # 座位数 seat_0 ~ seat_23
SEAT_PREFIX = "seat_"                 # 座位列名前缀
RANDOM_SEED = 42                      # 随机种子（要不要固定都行）

# ============================================================
#                   读入 & 基本预处理
# ============================================================
df = pd.read_csv(INPUT_CSV)

# 假设列名为 'timestamp' 和 'occupant_count'
# 解析时间（包含 +08:00 的时区信息）
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 过滤掉 occupant_count == 0 的行
df = df[df["occupant_count"] > 0].copy()

# 提取小时，筛选夜间（19:00 - 23:59 或 00:00 - 06:59）
df["hour"] = df["timestamp"].dt.hour
mask_night = (df["hour"] >= 19) | (df["hour"] < 7)
df = df[mask_night].copy()

# 按时间排序
df = df.sort_values("timestamp").reset_index(drop=True)

# ============================================================
#                生成座位分布（核心逻辑）
# ============================================================
rng = np.random.default_rng(RANDOM_SEED)

seat_cols = [f"seat_{i:02d}" for i in range(N_SEATS)]

records = []
current_occupied = set()   # 当前已坐的座位编号集合，例如 {0, 3, 5}
prev_count = None          # 上一个时刻的 occupant_count，用来比较增减

for idx, row in df.iterrows():
    timestamp = row["timestamp"]
    count = int(row["occupant_count"])

    # 如果 occupant_count 意外大于座位数，可以选择：截断 或 报错
    if count > N_SEATS:
        # 这里选择截断到 N_SEATS，你也可以改成 raise Exception
        count = N_SEATS

    # -------------------------------
    # 第一条记录 / 或前一时刻人数为 0 的情况：
    # 视为“第一批人随机坐”
    # -------------------------------
    if prev_count is None or prev_count == 0 or len(current_occupied) == 0:
        # 直接随机选 count 个座位
        chosen = rng.choice(N_SEATS, size=count, replace=False)
        current_occupied = set(chosen)

    else:
        diff = count - prev_count

        # 人数增加：在空座中随机加人
        if diff > 0:
            available = list(set(range(N_SEATS)) - current_occupied)
            add_n = min(diff, len(available))  # 防止空位不够
            if add_n > 0:
                new_seats = rng.choice(available, size=add_n, replace=False)
                current_occupied.update(new_seats)

        # 人数减少：从已有座位中随机减人
        elif diff < 0:
            remove_n = min(-diff, len(current_occupied))
            if remove_n > 0:
                remove_seats = rng.choice(list(current_occupied), size=remove_n, replace=False)
                current_occupied.difference_update(remove_seats)
        # diff == 0：人数不变，不动座位

    prev_count = count

    # 生成这一行的座位 0/1 数据
    seat_status = {
        f"{SEAT_PREFIX}{i}": (1 if i in current_occupied else 0)
        for i in range(N_SEATS)
    }

    record = {
        "timestamp": timestamp,
        "occupant_count": count,
        **seat_status
    }
    records.append(record)

# ============================================================
#               生成输出 DataFrame & 写 CSV
# ============================================================
out_df = pd.DataFrame(records)

# 插入 id 列，按顺序从 0 到 n-1
out_df.insert(1, "id", range(len(out_df)))

# 保存到 CSV
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"Done! 已生成 {len(out_df)} 行数据，保存到 {OUTPUT_CSV}")
