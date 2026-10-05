import json
import os


TRAIN_PATH = "data/train_split.json"
CORRECTION_PATH = "data/correction.json"
BACKUP_PATH = "data/train_split_backup.json"
OUTPUT_PATH = "data/train_v2.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_question(item):
    for message in item["messages"]:
        if message["role"] == "user":
            return message["content"].strip()
    return ""


# ==========================
# 读取原始训练集
# ==========================

train_data = load_json(TRAIN_PATH)
correction_data = load_json(CORRECTION_PATH)

print("原训练集：", len(train_data))
print("纠错数据：", len(correction_data))


# ==========================
# 备份原始训练集
# ==========================

with open(BACKUP_PATH, "w", encoding="utf-8") as f:
    json.dump(
        train_data,
        f,
        ensure_ascii=False,
        indent=2
    )

print("原训练集备份完成：", BACKUP_PATH)


# ==========================
# 去除完全重复的问题
# ==========================

existing_questions = {
    get_question(item)
    for item in train_data
}

added = 0
skipped = 0

for item in correction_data:

    question = get_question(item)

    if not question:
        print("发现没有用户问题的数据，跳过")
        skipped += 1
        continue

    if question in existing_questions:
        print("已存在，跳过：", question)
        skipped += 1
        continue

    train_data.append(item)
    existing_questions.add(question)

    added += 1


# ==========================
# 保存第二版训练集
# ==========================

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        train_data,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n==============================")
print("数据增强完成")
print("==============================")
print("新增数据：", added)
print("跳过数据：", skipped)
print("第二版训练集：", len(train_data))
print("保存位置：", OUTPUT_PATH)
print("原始训练集备份：", BACKUP_PATH)