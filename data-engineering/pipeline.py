"""
数据工程 Pipeline Demo — 展示预处理核心步骤

流程: 读取文本 → 语言检测 → 规则过滤 → MinHash 去重 → 输出

这只是演示。真实训练管线的规模是 TB 级 + 分布式处理。
"""
import re
import hashlib
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# 1. 规则过滤
# ═══════════════════════════════════════════════════════════
def rule_filter(doc: str, min_len=50, max_non_alpha_ratio=0.5) -> bool:
    """返回 True = 保留, False = 丢弃"""
    if len(doc) < min_len:
        return False

    # 非字母字符比例过高 → 可能是乱码或 HTML
    alpha_count = sum(1 for c in doc if c.isalpha())
    if alpha_count / max(len(doc), 1) < (1 - max_non_alpha_ratio):
        return False

    # 行数太少但总长很长 → 可能是没有断行的垃圾
    lines = doc.split('\n')
    if len(doc) > 1000 and len(lines) < 3:
        return False

    # 重复 n-gram 比例过高 → SEO 垃圾
    words = doc.split()
    if len(words) > 50:
        unique_bigrams = len(set(zip(words, words[1:])))
        if unique_bigrams / len(words) < 0.3:
            return False

    return True


# ═══════════════════════════════════════════════════════════
# 2. 语言检测（简化版 — 字符范围判断）
# ═══════════════════════════════════════════════════════════
def is_english(doc: str, threshold=0.7) -> bool:
    """统计拉丁字母占比，简单判断是否为英文"""
    latin = sum(1 for c in doc if c.isascii() and c.isalpha())
    total = sum(1 for c in doc if c.isalpha())
    return total > 0 and latin / total >= threshold


# ═══════════════════════════════════════════════════════════
# 3. MinHash 去重（简化版）
# ═══════════════════════════════════════════════════════════
def shingle(doc: str, k=3):
    """把文档拆成 k-shingles（字符级 n-gram）"""
    return set(doc[i:i+k] for i in range(len(doc) - k + 1))


def minhash_signature(shingles: set, num_hashes=64):
    """
    计算 MinHash 签名

    对每个 shingle 做 num_hashes 次哈希，每轮取最小的哈希值
    最终签名是一个长度为 num_hashes 的列表
    """
    sig = [float('inf')] * num_hashes
    for s in shingles:
        for i in range(num_hashes):
            h = hash((i, s)) & 0xFFFFFFFF
            if h < sig[i]:
                sig[i] = h
    return sig


def estimate_jaccard(sig1, sig2):
    """用两个 MinHash 签名的重合率估算 Jaccard 相似度"""
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def dedup_documents(docs, threshold=0.8):
    """
    用 MinHash 去重：相似度 > threshold 的只保留第一条
    复杂度 O(n²)，真实场景用 LSH 降到 O(n)
    """
    signatures = [minhash_signature(shingle(d)) for d in docs]
    keep = [True] * len(docs)

    for i in range(len(docs)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(docs)):
            if not keep[j]:
                continue
            if estimate_jaccard(signatures[i], signatures[j]) > threshold:
                keep[j] = False  # j 和 i 太像，去掉 j

    return [d for d, k in zip(docs, keep) if k]


# ═══════════════════════════════════════════════════════════
# 4. PII 脱敏（简化版）
# ═══════════════════════════════════════════════════════════
def mask_pii(doc: str) -> str:
    """用正则去掉常见的 PII"""
    doc = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  '[EMAIL]', doc)
    doc = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', doc)
    doc = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', doc)
    return doc


# ═══════════════════════════════════════════════════════════
# 5. 完整 Pipeline
# ═══════════════════════════════════════════════════════════
def process_documents(raw_docs, verbose=True):
    """
    完整的数据处理管道

    输入: 原始文档列表
    输出: 清洗后的文档列表
    """
    if verbose:
        print(f"原始文档数: {len(raw_docs)}")

    # Step 1: 语言过滤
    docs = [d for d in raw_docs if is_english(d)]
    if verbose:
        print(f"语言过滤后: {len(docs)}")

    # Step 2: 规则过滤
    docs = [d for d in docs if rule_filter(d)]
    if verbose:
        print(f"规则过滤后: {len(docs)}")

    # Step 3: PII 脱敏
    docs = [mask_pii(d) for d in docs]
    if verbose:
        print(f"PII 脱敏完成")

    # Step 4: MinHash 去重
    if len(docs) > 1:
        docs = dedup_documents(docs, threshold=0.8)
    if verbose:
        print(f"去重后: {len(docs)}")

    return docs


# ═══════════════════════════════════════════════════════════
# 6. 跑一下
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 造一些演示数据
    raw_docs = [
        # 1: 正常文本
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn and improve from experience without being explicitly "
        "programmed. Deep learning uses neural networks with many layers.",

        # 2: 太短 → 被规则过滤
        "Hi.",

        # 3: 正常文本
        "Python is a high-level programming language known for its readability "
        "and versatility. It is widely used in data science, web development, "
        "and artificial intelligence research across the world.",

        # 4: 乱码 → 被非字母比例过滤
        "<div>@@@@###!!!$$$%%%^^^&&&***((()))___---===+++[[[]]]]</div>",

        # 5: 重复的 SEO 垃圾 → 被 n-gram 比例过滤
        "buy cheap pills buy cheap pills buy cheap pills buy cheap pills "
        "buy cheap pills buy cheap pills buy cheap pills buy cheap pills",

        # 6: 和文档 1 高度相似 → 被 MinHash 去掉
        "Machine learning is a subset of artificial intelligence that allows "
        "systems to learn and improve from experience without explicit programming. "
        "Deep learning uses multi-layer neural networks.",

        # 7: 带 PII
        "Contact me at john.doe@example.com or call 555-123-4567 for more info "
        "about the machine learning course that starts next month.",
    ]

    cleaned = process_documents(raw_docs)

    print(f"\n{'='*60}")
    print(f"最终保留 {len(cleaned)} / {len(raw_docs)} 篇文档")
    print(f"{'='*60}\n")

    for i, doc in enumerate(cleaned):
        print(f"[文档 {i+1}] ({len(doc)} 字符)")
        print(doc[:150])
        print()
