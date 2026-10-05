import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TRAIN_FILE = DATA_DIR / "train_v3.json"
HOLDOUT_FILE = DATA_DIR / "holdout.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_train_questions(data):
    result = set()

    for item in data:
        for msg in item["messages"]:
            if msg["role"] == "user":
                result.add(msg["content"].strip())

    return result


def get_holdout_questions(data):
    return {
        item["question"].strip()
        for item in data
    }


print("=" * 60)
print("检查 V3 数据泄漏")
print("=" * 60)


train_data = load_json(TRAIN_FILE)
holdout_data = load_json(HOLDOUT_FILE)


train_questions = get_train_questions(train_data)
holdout_questions = get_holdout_questions(holdout_data)


same = train_questions & holdout_questions


print("V3训练问题数:", len(train_questions))
print("Holdout问题数:", len(holdout_questions))
print()


if same:
    print("⚠️ 发现泄漏")
    print("重复数量:", len(same))

    for q in same:
        print("-", q)

else:
    print("✅ 未发现数据泄漏")
    print("可以开始 V3 QLoRA训练")


print("=" * 60)