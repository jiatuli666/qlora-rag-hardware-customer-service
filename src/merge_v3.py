import json
from pathlib import Path

# =========================
# 路径
# =========================
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

V2_FILE = DATA_DIR / "train_v2.json"
V3_FILE = DATA_DIR / "v3_correction.json"
OUTPUT_FILE = DATA_DIR / "train_v3.json"
BACKUP_FILE = DATA_DIR / "train_v2_backup.json"


# =========================
# 读取 JSON
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 提取用户问题
# =========================
def get_question(item):
    for message in item.get("messages", []):
        if message.get("role") == "user":
            return message.get("content", "").strip()
    return ""


# =========================
# 主程序
# =========================
print("=" * 60)
print("开始合并 V2 + V3 纠错数据")
print("=" * 60)

v2_data = load_json(V2_FILE)
v3_data = load_json(V3_FILE)

print(f"V2训练集：{len(v2_data)} 条")
print(f"V3纠错数据：{len(v3_data)} 条")

# V2备份
with open(BACKUP_FILE, "w", encoding="utf-8") as f:
    json.dump(v2_data, f, ensure_ascii=False, indent=2)

# 建立已有问题集合
existing_questions = set()

for item in v2_data:
    question = get_question(item)

    if question:
        existing_questions.add(question)


# =========================
# 合并
# =========================
new_data = []
duplicate_data = []

for item in v3_data:
    question = get_question(item)

    if not question:
        print("⚠️ 发现没有 user 问题的数据，跳过")
        continue

    if question in existing_questions:
        duplicate_data.append(question)
    else:
        new_data.append(item)
        existing_questions.add(question)


final_data = v2_data + new_data


# =========================
# 保存
# =========================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)


# =========================
# 输出结果
# =========================
print()
print("=" * 60)
print("V3数据合并完成")
print("=" * 60)

print(f"原V2训练集：{len(v2_data)} 条")
print(f"V3纠错数据：{len(v3_data)} 条")
print(f"新增数据：{len(new_data)} 条")
print(f"重复数据：{len(duplicate_data)} 条")
print(f"最终V3训练集：{len(final_data)} 条")

print()
print(f"V2备份：{BACKUP_FILE}")
print(f"V3训练集：{OUTPUT_FILE}")

if duplicate_data:
    print()
    print("以下数据与V2重复：")
    for q in duplicate_data:
        print("-", q)

print()
print("=" * 60)