"""
MergeBERT Evaluation Script for Google Colab

使用方法:
1. 确保已上传以下文件到 Colab:
   - inferencecolab.py
   - mergebert.pt (训练好的模型权重)
   - autopsy_time20.json (测试数据)

2. 运行:
   !python eval.py

或在 notebook 中:
   %run eval.py
"""

import json
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict



# ==================== 配置 ====================
INPUT_FILE = "/content/spring-boot_time20.json"
OUTPUT_FILE = "/content/sis_time20eval.json"
TOP_K = 5  # 获取 top5 预测结果


def load_model_for_eval(model_path=MODEL_PATH, checkpoint_path=SAVED_MODEL_PATH):
    """加载模型用于评估"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Loaded tokenizer from: {model_path}")

    model = MergeBERTModel(encoder_path=model_path, num_labels=NUM_LABELS)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model checkpoint from: {checkpoint_path}")

    return model, tokenizer, device


def get_topk_predictions(model, tokenizer, device, a_text, b_text, o_text, k=5):
    """
    获取 top-k 预测结果

    返回:
        List[str]: top-k 个预测的解析结果文本
    """
    try:
        # Tokenize
        a_tokens = tokenizer.tokenize(a_text) if a_text else []
        b_tokens = tokenizer.tokenize(b_text) if b_text else []
        o_tokens = tokenizer.tokenize(o_text) if o_text else []

        # 如果所有内容都为空，返回空结果
        if not a_tokens and not b_tokens and not o_tokens:
            return [""] * k

        # Token-level merge
        hunks = token_level_merge(a_tokens, b_tokens, o_tokens)
        conflict_hunks = [h for h in hunks if isinstance(h, tuple)]

        # 如果没有冲突，直接返回合并结果
        if len(conflict_hunks) == 0:
            merged_tokens = hunks[0] if hunks else []
            merged_text = tokenizer.convert_tokens_to_string(merged_tokens)
            return [merged_text] * k  # 返回相同结果

        # 如果有多个冲突，暂时只处理第一个
        conflict_idx = -1
        for i, hunk in enumerate(hunks):
            if isinstance(hunk, tuple):
                conflict_idx = i
                break

        prefix_tokens = [token for hunk in hunks[:conflict_idx] for token in hunk]
        suffix_tokens = [token for hunk in hunks[conflict_idx+1:] for token in hunk]
        a_conflict, b_conflict, o_conflict = conflict_hunks[0]

        # Prepare data for model
        a_o_aligned, o_a_aligned, _ = align_and_get_edit_sequence(a_tokens, o_tokens)
        b_o_aligned, o_b_aligned, _ = align_and_get_edit_sequence(b_tokens, o_tokens)

        def process_sequence(seq):
            ids = tokenizer.convert_tokens_to_ids(seq)
            if len(ids) > MAX_LENGTH:
                ids = ids[:MAX_LENGTH]
            else:
                ids += [tokenizer.pad_token_id] * (MAX_LENGTH - len(ids))
            return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

        a_o_ids = process_sequence(a_o_aligned)
        o_a_ids = process_sequence(o_a_aligned)
        b_o_ids = process_sequence(b_o_aligned)
        o_b_ids = process_sequence(o_b_aligned)

        # Run inference and get top-k predictions
        with torch.no_grad():
            outputs = model(a_o_ids, o_a_ids, b_o_ids, o_b_ids)
            logits = outputs['logits']

            # Get top-k predictions
            probs = torch.softmax(logits, dim=1)
            topk_probs, topk_indices = torch.topk(probs, min(k, NUM_LABELS), dim=1)
            topk_indices = topk_indices[0].cpu().numpy()

        # Generate resolved text for each top-k prediction
        results = []
        for pred_idx in topk_indices:
            pattern = REVERSE_LABEL_MAP.get(int(pred_idx))
            if pattern:
                resolved_conflict = apply_pattern(pattern, a_conflict, b_conflict, o_conflict)
                if resolved_conflict is not None:
                    reconstructed_tokens = prefix_tokens + resolved_conflict + suffix_tokens
                    resolved_text = tokenizer.convert_tokens_to_string(reconstructed_tokens)
                    results.append(resolved_text)
                else:
                    results.append("")
            else:
                results.append("")

        # Pad to k results if needed
        while len(results) < k:
            results.append("")

        return results[:k]

    except Exception as e:
        print(f"Error during prediction: {e}")
        return [""] * k


def evaluate_dataset(input_file, output_file, model_path=MODEL_PATH, checkpoint_path=SAVED_MODEL_PATH):
    """评估整个数据集"""
    # 加载数据
    print(f"Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    # 加载模型
    model, tokenizer, device = load_model_for_eval(model_path, checkpoint_path)

    # 处理每个样本
    results = []
    for item in tqdm(data, desc="Evaluating"):
        # 提取输入
        a_text = item.get("a_contents", "")
        b_text = item.get("b_contents", "")
        o_text = item.get("o_contents", "")

        # 获取 top-k 预测
        answers = get_topk_predictions(model, tokenizer, device, a_text, b_text, o_text, k=TOP_K)

        # 创建结果对象（保留原始字段并添加 answers）
        result_item = item.copy()
        result_item["answers"] = answers
        results.append(result_item)

    # 保存结果
    print(f"Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Evaluation complete. Saved {len(results)} results.")

    return results


# ==================== 评估指标函数 ====================
def levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串之间的 Levenshtein 编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # 插入、删除、替换的代价
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def perfect_match_rate(ours: str, target: str) -> float:
    """
    计算两个字符串的相似度（基于 Levenshtein 距离）

    参数:
        ours: 预测结果
        target: 目标结果

    返回:
        相似度百分比 (0-100)
    """
    import re
    # 移除所有空白字符
    ours = re.sub(r'\s*', '', ours) if ours else ''
    target = re.sub(r'\s*', '', target) if target else ''

    if len(ours) == 0 and len(target) == 0:
        return 100.0

    # 计算 Levenshtein 距离
    distance = levenshtein_distance(ours, target)

    # 计算相似度
    max_len = max(len(ours), len(target))
    similarity = 1 - distance / max_len if max_len > 0 else 1.0

    # 返回百分比形式，保留两位小数
    return round(similarity * 100, 2)


def is_perfect_match(cand: str, resolution: str, threshold: float = 100.0) -> bool:
    """判断是否为完美匹配（相似度达到阈值）"""
    return perfect_match_rate(cand, resolution) >= threshold


def count_topk_perfect_match(data: List[dict], answer_key: str = "answers") -> Dict[str, float]:
    """计算 top-k 完全匹配率"""
    result = {
        "total": len(data),
        "top1_perfect": 0,
        "top3_perfect": 0,
        "top5_perfect": 0
    }

    for obj in data:
        resolution = obj.get("res_region", "").strip()
        answers = obj.get(answer_key, [])

        top1 = top3 = top5 = False

        for k, cand in enumerate(answers[:5]):
            if is_perfect_match(cand, resolution):
                if k == 0:
                    top1 = True
                if k < 3:
                    top3 = True
                top5 = True
                break

        result["top1_perfect"] += int(top1)
        result["top3_perfect"] += int(top3 or top1)
        result["top5_perfect"] += int(top5 or top3 or top1)

    total = result["total"]
    if total > 0:
        result["top1_rate_%"] = result["top1_perfect"] / total * 100
        result["top3_rate_%"] = result["top3_perfect"] / total * 100
        result["top5_rate_%"] = result["top5_perfect"] / total * 100
    else:
        result["top1_rate_%"] = 0
        result["top3_rate_%"] = 0
        result["top5_rate_%"] = 0

    return result


def main():
    """主函数"""
    # 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        print("Please upload autopsy_time20.json to /content/")
        return

    # 检查模型文件
    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"Error: Model checkpoint not found: {SAVED_MODEL_PATH}")
        print("Please upload mergebert.pt to /content/saved_models/")
        return

    # 运行评估
    results = evaluate_dataset(INPUT_FILE, OUTPUT_FILE)

    # 计算指标
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)

    metrics = count_topk_perfect_match(results, "answers")

    print(f"Total samples: {metrics['total']}")
    print(f"Top-1 Perfect Match: {metrics['top1_perfect']} ({metrics['top1_rate_%']:.2f}%)")
    print(f"Top-3 Perfect Match: {metrics['top3_perfect']} ({metrics['top3_rate_%']:.2f}%)")
    print(f"Top-5 Perfect Match: {metrics['top5_perfect']} ({metrics['top5_rate_%']:.2f}%)")
    print("="*50)

    return metrics


if __name__ == "__main__":
    main()
