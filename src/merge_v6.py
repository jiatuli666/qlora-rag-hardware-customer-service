import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

V5_FILE = BASE_DIR / "data" / "train_v5.json"
V6_FILE = BASE_DIR / "data" / "v6_correction.json"
OUTPUT_FILE = BASE_DIR / "data" / "train_v6.json"


def get_question(item):
    return item["messages"][1]["content"].strip()


with open(V5_FILE, "r", encoding="utf-8") as f:
    v5_data = json.load(f)

with open(V6_FILE, "r", encoding="utf-8") as f:
    v6_data = json.load(f)


print("=" * 60)
print("V6训练数据合并")
print("=" * 60)

print("V5训练数据:", len(v5_data))
print("V6纠错数据:", len(v6_data))


v5_questions = {
    get_question(item)
    for item in v5_data
}

v6_questions = set()

new_data = []
duplicate_count = 0


for item in v6_data:

    question = get_question(item)

    if question in v5_questions:
        duplicate_count += 1
        continue

    if question in v6_questions:
        duplicate_count += 1
        continue

    v6_questions.add(question)
    new_data.append(item)


final_data = v5_data + new_data


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        final_data,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("=" * 60)
print("V6合并完成")
print("=" * 60)

print("V5训练数据:", len(v5_data))
print("V6原始数据:", len(v6_data))
print("V6新增数据:", len(new_data))
print("重复数据:", duplicate_count)
print("最终V6训练数据:", len(final_data))

print()
print("保存位置:")
print(OUTPUT_FILE)