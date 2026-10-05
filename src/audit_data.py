import json
import re
from collections import Counter


INPUT_FILE = "data/train_v6.json"


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_question(item):
    for msg in item.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def get_answer(item):
    for msg in item.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "").strip()
    return ""


def check_malformed(answer):
    problems = []

    # 1. 明显残缺
    bad_patterns = [
        "为。",
        "是。",
        "不。",
        "配。",
        "了。",
        "的。",
        "不。",
        "是否正确装入",
        "具体以商品规格为。",
        "两者槽型不。",
        "规格配。",
    ]

    for p in bad_patterns:
        if p in answer:
            problems.append(f"疑似残缺：{p}")

    # 2. 连续重复词
    if re.search(r"(打滑){2,}", answer):
        problems.append("连续重复：打滑")

    if re.search(r"(检查){3,}", answer):
        problems.append("连续重复：检查")

    if re.search(r"(是否){3,}", answer):
        problems.append("连续重复：是否")

    # 3. 连续重复句子
    sentences = re.split(r"[。！？]", answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    for i in range(len(sentences) - 1):
        if sentences[i] == sentences[i + 1]:
            problems.append("连续重复句子")

    # 4. 超长重复
    words = re.findall(r"[\u4e00-\u9fff]+", answer)
    if len(words) > 10:
        counter = Counter(words)
        most_common_word, count = counter.most_common(1)[0]

        if count >= 5:
            problems.append(
                f"高频重复词：{most_common_word} × {count}"
            )

    return problems
def check_malformed(answer):
    problems = []

    answer = answer.strip()

    # ==============================
    # 1. 真正的残句检查
    # ==============================

    bad_exact_endings = [
        "规格匹。",
        "具体以商品规格为。",
        "商品规格为。",
        "根据实际规格为。",
        "具体配置为。",
        "规格配。",
        "两者槽型不。",
    ]

    for p in bad_exact_endings:
        if answer.endswith(p):
            problems.append(f"疑似残句：{p}")

    # ==============================
    # 2. 明显异常空格
    # ==============================

    if "商 品" in answer:
        problems.append("存在异常空格：商 品")

    if "要 以" in answer:
        problems.append("存在异常空格：要 以")

    # ==============================
    # 3. 连续重复
    # ==============================

    if "打滑打滑" in answer:
        problems.append("重复词：打滑")

    if "检查检查" in answer:
        problems.append("重复词：检查")

    if "是否是否" in answer:
        problems.append("重复词：是否")

    # ==============================
    # 4. 连续重复句子
    # ==============================

    sentences = re.split(r"[。！？]", answer)

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    for i in range(len(sentences) - 1):

        if sentences[i] == sentences[i + 1]:
            problems.append(
                f"连续重复句子：{sentences[i]}"
            )

    # ==============================
    # 5. 极端重复
    # ==============================

    if len(answer) > 80:

        # 同一个短语连续出现
        if re.search(
            r"(.{4,12})\1{2,}",
            answer
        ):
            problems.append(
                "存在连续重复短语"
            )

    return problems

def main():
    data = load_data(INPUT_FILE)

    print("=" * 60)
    print("QLoRA V6 数据质量审计")
    print("=" * 60)

    print(f"数据总量：{len(data)}")

    # ------------------------------------------------
    # 1. 检查重复问题
    # ------------------------------------------------
    questions = [get_question(x) for x in data]

    question_counter = Counter(questions)

    duplicate_questions = {
        q: count
        for q, count in question_counter.items()
        if q and count > 1
    }

    print("\n【1. 重复问题】")

    if duplicate_questions:
        for q, count in duplicate_questions.items():
            print(f"重复 {count} 次：{q}")
    else:
        print("✅ 没有发现完全重复的问题")

    # ------------------------------------------------
    # 2. 检查空问题 / 空答案
    # ------------------------------------------------
    print("\n【2. 空数据检查】")

    empty_count = 0

    for i, item in enumerate(data):
        q = get_question(item)
        a = get_answer(item)

        if not q:
            print(f"❌ 第 {i + 1} 条：问题为空")
            empty_count += 1

        if not a:
            print(f"❌ 第 {i + 1} 条：答案为空")
            empty_count += 1

    if empty_count == 0:
        print("✅ 没有发现空问题或空答案")

    # ------------------------------------------------
    # 3. 检查异常答案
    # ------------------------------------------------
    print("\n【3. 异常答案检查】")

    bad_count = 0

    for i, item in enumerate(data):
        q = get_question(item)
        a = get_answer(item)

        problems = check_malformed(a)

        if problems:
            bad_count += 1

            print("\n" + "-" * 60)
            print(f"第 {i + 1} 条")
            print("问题：", q)
            print("答案：", a)

            for p in problems:
                print("⚠️", p)

    if bad_count == 0:
        print("✅ 没有发现明显异常答案")

    # ------------------------------------------------
    # 4. PH2 数据检查
    # ------------------------------------------------
    print("\n【4. PH2 数据检查】")

    ph2_data = []

    for item in data:
        q = get_question(item)
        a = get_answer(item)

        if "PH2" in q or "PH2" in a:
            ph2_data.append((q, a))

    print(f"PH2相关数据：{len(ph2_data)} 条")

    for q, a in ph2_data:
        if "2毫米" in a or "2mm" in a:
            print("\n⚠️ 可能存在 PH2 = 2mm 的错误")
            print("问题：", q)
            print("答案：", a)

    # ------------------------------------------------
    # 5. S2 数据检查
    # ------------------------------------------------
    print("\n【5. S2 数据检查】")

    s2_data = []

    for item in data:
        q = get_question(item)
        a = get_answer(item)

        if "S2" in q or "S2" in a:
            s2_data.append((q, a))

    print(f"S2相关数据：{len(s2_data)} 条")

    for q, a in s2_data:
        if "属于不锈钢" in a or "是不锈钢" in a:
            print("\n⚠️ 可能存在 S2 = 不锈钢 的错误")
            print("问题：", q)
            print("答案：", a)

    # ------------------------------------------------
    # 6. 长度计算检查
    # ------------------------------------------------
    print("\n【6. 长度计算数据检查】")

    calc_data = []

    for item in data:
        q = get_question(item)
        a = get_answer(item)

        if (
            "200mm" in q
            or "200 - 65" in q
            or "200和65" in q
            or "65mm" in q and "200mm" in q
        ):
            calc_data.append((q, a))

    for q, a in calc_data:
        print("\n问题：", q)
        print("答案：", a)

    print("\n" + "=" * 60)
    print("审计完成")
    print("=" * 60)


if __name__ == "__main__":
    main()