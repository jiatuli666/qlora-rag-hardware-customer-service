import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

V4_FILE = BASE_DIR / "data" / "train_v4.json"
V5_FILE = BASE_DIR / "data" / "v5_correction.json"
OUTPUT_FILE = BASE_DIR / "data" / "train_v5.json"


def get_question(item):
    return item["messages"][1]["content"].strip()


# ============================================================
# 读取数据
# ============================================================

with open(V4_FILE, "r", encoding="utf-8") as f:
    v4_data = json.load(f)

with open(V5_FILE, "r", encoding="utf-8") as f:
    v5_data = json.load(f)


print("=" * 60)
print("V5训练数据合并")
print("=" * 60)

print("V4训练数据:", len(v4_data))
print("V5纠错数据:", len(v5_data))


# ============================================================
# 建立已有问题集合
# ============================================================

v4_questions = {
    get_question(item)
    for item in v4_data
}

v5_questions = set()

new_data = []
duplicate_count = 0


# ============================================================
# 去重
# ============================================================

for item in v5_data:

    question = get_question(item)

    if question in v4_questions:
        duplicate_count += 1
        continue

    if question in v5_questions:
        duplicate_count += 1
        continue

    v5_questions.add(question)
    new_data.append(item)


# ============================================================
# 合并
# ============================================================

final_data = v4_data + new_data


# ============================================================
# 保存
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_data,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 输出结果
# ============================================================

print()
print("=" * 60)
print("V5合并完成")
print("=" * 60)

print("V4训练数据:", len(v4_data))
print("V5原始数据:", len(v5_data))
print("V5新增数据:", len(new_data))
print("重复数据:", duplicate_count)
print("最终V5训练数据:", len(final_data))

print()
print("保存位置:")
print(OUTPUT_FILE)