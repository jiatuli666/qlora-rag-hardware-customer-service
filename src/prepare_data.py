import json
import random
from pathlib import Path


# =========================
# 1. 设置文件路径
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT    / "data"

INPUT_FILE = DATA_DIR / "train.json"

TRAIN_FILE = DATA_DIR / "train_split.json"
VAL_FILE = DATA_DIR / "val.json"
TEST_FILE = DATA_DIR / "test.json"


# =========================
# 2. 读取原始数据
# =========================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


print("原始数据数量:", len(data))


# =========================
# 3. 检查数据格式
# =========================

for i, item in enumerate(data):

    if "messages" not in item:
        raise ValueError(f"第 {i + 1} 条数据缺少 messages")

    if not isinstance(item["messages"], list):
        raise ValueError(f"第 {i + 1} 条数据的 messages 不是列表")

    if len(item["messages"]) != 3:
        raise ValueError(f"第 {i + 1} 条数据应该包含 system、user、assistant")


print("数据格式检查通过")


# =========================
# 4. 打乱数据
# =========================

random.seed(42)

random.shuffle(data)


# =========================
# 5. 划分数据集
# =========================

total = len(data)

train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_data = data[:train_end]

val_data = data[train_end:val_end]

test_data = data[val_end:]


# =========================
# 6. 保存数据
# =========================

def save_json(file_path, dataset):

    with open(file_path, "w", encoding="utf-8") as f:

        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2
        )


save_json(TRAIN_FILE, train_data)

save_json(VAL_FILE, val_data)

save_json(TEST_FILE, test_data)


# =========================
# 7. 输出结果
# =========================

print()
print("============================")
print("数据集划分完成")
print("============================")

print("训练集:", len(train_data))

print("验证集:", len(val_data))

print("测试集:", len(test_data))

print()
print("文件位置:")

print(TRAIN_FILE)

print(VAL_FILE)

print(TEST_FILE)