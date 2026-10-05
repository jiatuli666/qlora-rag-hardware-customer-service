import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

V3_FILE = DATA_DIR / "train_v3.json"
V4_FILE = DATA_DIR / "v4_correction.json"
HOLDOUT_FILE = DATA_DIR / "holdout.json"

OUTPUT_FILE = DATA_DIR / "train_v4.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_question(item):
    for msg in item.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


print("=" * 60)
print("V4数据合并与Holdout泄漏检查")
print("=" * 60)


v3_data = load_json(V3_FILE)
v4_data = load_json(V4_FILE)
holdout_data = load_json(HOLDOUT_FILE)


# --------------------------------------------------
# 提取问题
# --------------------------------------------------

v3_questions = {
    get_question(item)
    for item in v3_data
    if get_question(item)
}

v4_questions = {
    get_question(item)
    for item in v4_data
    if get_question(item)
}

holdout_questions = {
    item["question"].strip()
    for item in holdout_data
}


# --------------------------------------------------
# V4内部重复
# --------------------------------------------------

v4_list = list(v4_questions)

if len(v4_list) != len(v4_data):
    print("⚠️ V4内部存在重复问题或空问题")


# --------------------------------------------------
# V4与V3重复
# --------------------------------------------------

v3_v4_duplicate = v3_questions & v4_questions


# --------------------------------------------------
# V4与Holdout重复
# --------------------------------------------------

v4_holdout_leak = v4_questions & holdout_questions


# --------------------------------------------------
# V3与Holdout重复
# --------------------------------------------------

v3_holdout_leak = v3_questions & holdout_questions


print()
print("V3训练数据:", len(v3_data))
print("V4纠错数据:", len(v4_data))
print("Holdout数据:", len(holdout_data))

print()
print("V3问题数量:", len(v3_questions))
print("V4问题数量:", len(v4_questions))
print("Holdout问题数量:", len(holdout_questions))


# --------------------------------------------------
# 输出重复
# --------------------------------------------------

print()
print("-" * 60)
print("V3 / V4重复:", len(v3_v4_duplicate))

if v3_v4_duplicate:
    for q in v3_v4_duplicate:
        print("  ", q)


print()
print("-" * 60)
print("V4 / Holdout泄漏:", len(v4_holdout_leak))

if v4_holdout_leak:
    for q in v4_holdout_leak:
        print("  ", q)


print()
print("-" * 60)
print("V3 / Holdout泄漏:", len(v3_holdout_leak))

if v3_holdout_leak:
    for q in v3_holdout_leak:
        print("  ", q)


# --------------------------------------------------
# 只有没有Holdout泄漏时才合并
# --------------------------------------------------

if v4_holdout_leak:
    print()
    print("❌ 检测到Holdout泄漏")
    print("暂不生成最终V4训练集")

else:

    # 去掉与V3完全重复的数据
    new_v4_data = [
        item
        for item in v4_data
        if get_question(item) not in v3_questions
    ]

    final_data = v3_data + new_v4_data

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            final_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 60)
    print("V4数据合并完成")
    print("=" * 60)

    print("V3原训练集:", len(v3_data))
    print("V4新增数据:", len(new_v4_data))
    print("最终V4训练集:", len(final_data))

    print()
    print("保存位置:")
    print(OUTPUT_FILE)