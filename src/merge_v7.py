import json
import shutil
from pathlib import Path


V6_FILE = Path("data/train_v6.json")
V7_CORRECTION_FILE = Path("data/v7_correction.json")
OUTPUT_FILE = Path("data/train_v7.json")
BACKUP_FILE = Path("data/train_v6_backup.json")


def get_question(item):
    for msg in item.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def main():

    # ==============================
    # 1. 检查文件
    # ==============================

    if not V6_FILE.exists():
        print("❌ 找不到 train_v6.json")
        return

    if not V7_CORRECTION_FILE.exists():
        print("❌ 找不到 v7_correction.json")
        return

    # ==============================
    # 2. 读取数据
    # ==============================

    with open(V6_FILE, "r", encoding="utf-8") as f:
        v6_data = json.load(f)

    with open(V7_CORRECTION_FILE, "r", encoding="utf-8") as f:
        v7_data = json.load(f)

    print("=" * 60)
    print("QLoRA V7 数据合并")
    print("=" * 60)

    print(f"V6训练数据：{len(v6_data)}")
    print(f"V7纠错数据：{len(v7_data)}")

    # ==============================
    # 3. 建立 V6 问题索引
    # ==============================

    existing_questions = set()

    for item in v6_data:

        q = get_question(item)

        if q:
            existing_questions.add(q)

    # ==============================
    # 4. 合并
    # ==============================

    new_data = []
    duplicate_data = []

    for item in v7_data:

        q = get_question(item)

        if not q:
            print("⚠️ 发现空问题，跳过")
            continue

        if q in existing_questions:

            duplicate_data.append(q)

        else:

            new_data.append(item)
            existing_questions.add(q)

    final_data = v6_data + new_data

    # ==============================
    # 5. 备份 V6
    # ==============================

    if not BACKUP_FILE.exists():

        shutil.copy2(
            V6_FILE,
            BACKUP_FILE
        )

        print(
            f"\n✅ 已备份 V6：{BACKUP_FILE}"
        )

    # ==============================
    # 6. 保存 V7
    # ==============================

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

    # ==============================
    # 7. 输出结果
    # ==============================

    print("\n" + "=" * 60)
    print("合并结果")
    print("=" * 60)

    print(f"V6原始数据：{len(v6_data)}")
    print(f"V7纠错数据：{len(v7_data)}")
    print(f"新增数据：{len(new_data)}")
    print(f"重复数据：{len(duplicate_data)}")
    print(f"最终V7数据：{len(final_data)}")

    # ==============================
    # 8. 输出重复问题
    # ==============================

    if duplicate_data:

        print("\n【重复问题】")

        for q in duplicate_data:

            print(
                f"⚠️ {q}"
            )

    else:

        print(
            "\n✅ V7纠错数据没有与V6重复的问题"
        )

    print("\n" + "=" * 60)
    print(
        f"✅ 已生成：{OUTPUT_FILE}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()