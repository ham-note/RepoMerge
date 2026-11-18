"""
MergeBERT Inference Script for Google Colab

使用方法:
1. 上传以下文件到 Colab 的 /content/ 目录:
   - inferencecolab.py (本文件)
   - mergebert.pt (训练好的模型权重)

2. 在 Colab 中运行以下命令安装依赖:
   !pip install torch transformers GitPython

3. 准备冲突文件 (三个版本):
   - file_a.txt (你的版本)
   - file_b.txt (他们的版本)
   - file_o.txt (共同祖先/基础版本)

4. 运行推理:
   方式1: 命令行
   !python inferencecolab.py --file_a file_a.txt --file_b file_b.txt --file_o file_o.txt

   方式2: 在 notebook 中调用函数
   from inferencecolab import resolve_conflict
   result = resolve_conflict("代码A内容", "代码B内容", "基础版本内容")
   print(result)
"""

import torch
import argparse
import os
import re
import shutil
import numpy as np
from pathlib import Path
from typing import Callable, List
from transformers import AutoTokenizer, RobertaModel, RobertaConfig
import torch.nn as nn

# ==================== Colab 路径配置 ====================
# 使用 Hugging Face 在线模型
MODEL_PATH = "huggingface/CodeBERTa-small-v1"
# 训练好的模型权重路径
SAVED_MODEL_PATH = "/content/saved_models/mergebert.pt"
MAX_LENGTH = 512
NUM_LABELS = 9

# 临时 git 仓库路径
TEMP_REPO_BASE_PATH = Path("/tmp/mergebert_repo_temp")

# ==================== 标签映射 ====================
REVERSE_LABEL_MAP = {
    0: "select_a",
    1: "select_b",
    2: "select_o",
    3: "concat_ab",
    4: "concat_ba",
    5: "select_a_del_o",
    6: "select_b_del_o",
    7: "concat_ab_del_o",
    8: "concat_ba_del_o",
}


# ==================== 模型定义 (内嵌 model.py) ====================
class MergeBERTModel(nn.Module):
    """MergeBERT model for merge conflict resolution."""
    def __init__(self, encoder_path, num_labels=9):
        super(MergeBERTModel, self).__init__()
        self.num_labels = num_labels

        config = RobertaConfig.from_pretrained(encoder_path)
        self.encoder = RobertaModel.from_pretrained(encoder_path, config=config)

        # Aggregation weights (learnable)
        self.aggregation_weights = nn.Parameter(torch.ones(4))

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, a_o_ids, o_a_ids, b_o_ids, o_b_ids, labels=None):
        output_ao = self.encoder(input_ids=a_o_ids).pooler_output
        output_oa = self.encoder(input_ids=o_a_ids).pooler_output
        output_bo = self.encoder(input_ids=b_o_ids).pooler_output
        output_ob = self.encoder(input_ids=o_b_ids).pooler_output

        agg_weights = nn.functional.softmax(self.aggregation_weights, dim=0)

        aggregated_output = (
            agg_weights[0] * output_ao +
            agg_weights[1] * output_oa +
            agg_weights[2] * output_bo +
            agg_weights[3] * output_ob
        )

        logits = self.classifier(aggregated_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits
        }


# ==================== 工具函数 (内嵌 utils.py 关键部分) ====================
edit_seq_tokens = {
    "eql": "=", "add": "+", "del": "-", "rep": "↔", "padding": "[PAD]"
}

def normalize_code(text: str) -> str:
    """Removes indents, empty lines, and duplicate whitespace."""
    lines = text.split('\n')
    processed_lines = [line.strip() for line in lines if line.strip()]
    return re.sub(r'\s+', ' ', " ".join(processed_lines))

def apply_pattern(pattern, a, b, o):
    if pattern == "select_a": return a
    if pattern == "select_b": return b
    if pattern == "select_o": return o
    if pattern == "concat_ab": return a + b
    if pattern == "concat_ba": return b + a
    o_set = set(o)
    if pattern == "select_a_del_o": return [token for token in a if token not in o_set]
    if pattern == "select_b_del_o": return [token for token in b if token not in o_set]
    if pattern == "concat_ab_del_o": return [token for token in a + b if token not in o_set]
    if pattern == "concat_ba_del_o": return [token for token in b + a if token not in o_set]
    return None

class OffsetRange:
    def __init__(self, start: int, end: int):
        self.start, self.end = start, end
    def __len__(self): return self.end - self.start

class SequenceDiff:
    def __init__(self, seq1Range: OffsetRange, seq2Range: OffsetRange):
        self.seq1Range, self.seq2Range = seq1Range, seq2Range

def compute(sequence1: List[str], sequence2: List[str], equalityScore: Callable[[int, int], float] = lambda x, y: 1 if x == y else 0) -> List[SequenceDiff]:
    if not sequence1 or not sequence2: return [SequenceDiff(OffsetRange(0, len(sequence1)), OffsetRange(0, len(sequence2)))]
    lcsLengths, directions, lengths = np.zeros((len(sequence1), len(sequence2))), np.zeros((len(sequence1), len(sequence2))), np.zeros((len(sequence1), len(sequence2)))
    for s1 in range(len(sequence1)):
        for s2 in range(len(sequence2)):
            h_len, v_len = (lcsLengths[s1 - 1, s2] if s1 > 0 else 0), (lcsLengths[s1, s2 - 1] if s2 > 0 else 0)
            ext_score = -1
            if sequence1[s1] == sequence2[s2]:
                ext_score = (lcsLengths[s1 - 1, s2 - 1] if s1 > 0 and s2 > 0 else 0) + equalityScore(sequence1[s1], sequence2[s2])
                if s1 > 0 and s2 > 0 and directions[s1 - 1, s2 - 1] == 3: ext_score += lengths[s1 - 1, s2 - 1]
            newValue = max(h_len, v_len, ext_score)
            if newValue == ext_score: lengths[s1, s2], directions[s1, s2] = (lengths[s1 - 1, s2 - 1] if s1 > 0 and s2 > 0 else 0) + 1, 3
            elif newValue == h_len: lengths[s1, s2], directions[s1, s2] = 0, 1
            else: lengths[s1, s2], directions[s1, s2] = 0, 2
            lcsLengths[s1, s2] = newValue
    result, s1, s2, last_s1, last_s2 = [], len(sequence1) - 1, len(sequence2) - 1, len(sequence1), len(sequence2)
    while s1 >= 0 and s2 >= 0:
        if directions[s1, s2] == 3:
            if s1 + 1 != last_s1 or s2 + 1 != last_s2: result.append(SequenceDiff(OffsetRange(s1 + 1, last_s1), OffsetRange(s2 + 1, last_s2)))
            last_s1, last_s2 = s1, s2
            s1, s2 = s1 - 1, s2 - 1
        elif directions[s1, s2] == 1: s1 -= 1
        else: s2 -= 1
    if -1 + 1 != last_s1 or -1 + 1 != last_s2: result.append(SequenceDiff(OffsetRange(0, last_s1), OffsetRange(0, last_s2)))
    result.reverse()
    return result

def get_edit_sequence(before: List[str], after: List[str]) -> List[List[str]]:
    ess, seq, before_padded, after_padded, b_trav, a_trav = compute(before, after), [], [], [], 0, 0
    for es in ess:
        b_start, b_end, a_start, a_end, b_len, a_len = es.seq1Range.start, es.seq1Range.end, es.seq2Range.start, es.seq2Range.end, len(es.seq1Range), len(es.seq2Range)
        seq.extend([edit_seq_tokens["eql"]] * (b_start - b_trav)); before_padded.extend(before[b_trav:b_start]); after_padded.extend(after[a_trav:a_start])
        if b_len == 0: seq.extend([edit_seq_tokens["add"]] * a_len); before_padded.extend([edit_seq_tokens["padding"]] * a_len); after_padded.extend(after[a_start:a_end])
        elif a_len == 0: seq.extend([edit_seq_tokens["del"]] * b_len); before_padded.extend(before[b_start:b_end]); after_padded.extend([edit_seq_tokens["padding"]] * b_len)
        else:
            if b_len > a_len: seq.extend([edit_seq_tokens["rep"]] * a_len + [edit_seq_tokens["del"]] * (b_len - a_len)); before_padded.extend(before[b_start:b_end]); after_padded.extend(after[a_start:a_end] + [edit_seq_tokens["padding"]] * (b_len - a_len))
            else: seq.extend([edit_seq_tokens["rep"]] * b_len + [edit_seq_tokens["add"]] * (a_len - b_len)); after_padded.extend(after[a_start:a_end]); before_padded.extend(before[b_start:b_end] + [edit_seq_tokens["padding"]] * (a_len - b_len))
        b_trav, a_trav = b_end, a_end
    seq.extend([edit_seq_tokens["eql"]] * (len(before) - b_trav)); before_padded.extend(before[b_trav:]); after_padded.extend(after[a_trav:])
    return seq, before_padded, after_padded

def align_and_get_edit_sequence(before: List[str], after: List[str]) -> List[List[str]]:
    return get_edit_sequence(before, after)

def write_tokens_to_file(tokens, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for token in tokens: f.write(token + '\n')

def read_tokens_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f: return [line.strip('\n') for line in f.readlines()]

def token_level_merge(a_tokens, b_tokens, o_tokens):
    try:
        from git import Repo, Git
    except ImportError:
        raise ImportError("请先安装 GitPython: !pip install GitPython")

    temp_repo_path = TEMP_REPO_BASE_PATH / str(os.getpid())
    if temp_repo_path.exists(): shutil.rmtree(temp_repo_path)
    repo = Repo.init(temp_repo_path, initial_branch='master'); _git = Git(temp_repo_path); _git.config("merge.conflictstyle", "diff3"); tmp_file = temp_repo_path / "tokens.txt"
    write_tokens_to_file(o_tokens, tmp_file); _git.add('.'); repo.index.commit("base")
    _git.branch('a_branch'); _git.checkout('a_branch'); write_tokens_to_file(a_tokens, tmp_file); _git.add('.'); repo.index.commit("commit a")
    _git.checkout('master'); _git.branch('b_branch'); _git.checkout('b_branch'); write_tokens_to_file(b_tokens, tmp_file); _git.add('.'); repo.index.commit("commit b")
    try: _git.merge('a_branch'); return [read_tokens_from_file(tmp_file)]
    except Exception: return parse_conflict_markers(tmp_file.read_text())

def parse_conflict_markers(content):
    hunks, lines, current_hunk, i = [], content.split('\n'), [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('<<<<<<<'):
            if current_hunk: hunks.append([h for h in current_hunk if h]); current_hunk = []
            a_conflict, o_conflict, b_conflict = [], [], []; i += 1
            while not lines[i].startswith('|||||||'): a_conflict.append(lines[i]); i += 1
            i += 1
            while not lines[i].startswith('======='): o_conflict.append(lines[i]); i += 1
            i += 1
            while not lines[i].startswith('>>>>>>>'): b_conflict.append(lines[i]); i += 1
            hunks.append((a_conflict, b_conflict, o_conflict))
        else: current_hunk.append(line)
        i += 1
    if current_hunk: current_hunk = [h for h in current_hunk if h];
    if current_hunk: hunks.append(current_hunk)
    return hunks


# ==================== 全局变量 (用于缓存模型) ====================
_cached_model = None
_cached_tokenizer = None
_cached_device = None


def load_model(model_path=MODEL_PATH, checkpoint_path=SAVED_MODEL_PATH, device=None):
    """加载模型 (支持缓存，避免重复加载)"""
    global _cached_model, _cached_tokenizer, _cached_device

    if _cached_model is not None:
        return _cached_model, _cached_tokenizer, _cached_device

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading model from: {model_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = MergeBERTModel(encoder_path=model_path, num_labels=NUM_LABELS)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_device = device

    return model, tokenizer, device


def resolve_conflict(a_text: str, b_text: str, o_text: str,
                     model_path=MODEL_PATH,
                     checkpoint_path=SAVED_MODEL_PATH) -> str:
    """
    解决合并冲突的主函数

    参数:
        a_text: 版本 A 的代码 (你的版本)
        b_text: 版本 B 的代码 (他们的版本)
        o_text: 基础版本的代码 (共同祖先)
        model_path: 预训练模型路径
        checkpoint_path: 训练好的权重路径

    返回:
        resolved_text: 解决冲突后的代码
    """
    model, tokenizer, device = load_model(model_path, checkpoint_path)

    # Tokenize
    a_tokens = tokenizer.tokenize(a_text)
    b_tokens = tokenizer.tokenize(b_text)
    o_tokens = tokenizer.tokenize(o_text)

    # Token-level merge
    hunks = token_level_merge(a_tokens, b_tokens, o_tokens)
    conflict_hunks = [h for h in hunks if isinstance(h, tuple)]

    if len(conflict_hunks) == 0:
        print("No conflict detected at token level. Returning merged result.")
        merged_tokens = hunks[0] if hunks else []
        return tokenizer.convert_tokens_to_string(merged_tokens)

    if len(conflict_hunks) > 1:
        raise ValueError(f"Found {len(conflict_hunks)} conflicts. This model only supports single-conflict resolution.")

    # Find conflict index
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

    # Run inference
    with torch.no_grad():
        outputs = model(a_o_ids, o_a_ids, b_o_ids, o_b_ids)
        logits = outputs['logits']
        prediction = torch.argmax(logits, dim=1).item()

    # Decode and reconstruct
    predicted_pattern = REVERSE_LABEL_MAP.get(prediction)
    if not predicted_pattern:
        raise ValueError(f"Unknown pattern predicted with ID: {prediction}")

    print(f"Predicted resolution pattern: {predicted_pattern}")

    resolved_conflict = apply_pattern(predicted_pattern, a_conflict, b_conflict, o_conflict)
    reconstructed_tokens = prefix_tokens + resolved_conflict + suffix_tokens
    resolved_text = tokenizer.convert_tokens_to_string(reconstructed_tokens)

    return resolved_text


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="MergeBERT Inference for Colab")
    parser.add_argument("--file_a", type=str, required=True, help="Path to file A (yours)")
    parser.add_argument("--file_b", type=str, required=True, help="Path to file B (theirs)")
    parser.add_argument("--file_o", type=str, required=True, help="Path to file O (base)")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Path to pretrained CodeBERTa model")
    parser.add_argument("--checkpoint", type=str, default=SAVED_MODEL_PATH,
                        help="Path to trained mergebert.pt checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save resolved output (optional)")
    args = parser.parse_args()

    # 检查文件是否存在
    for path, name in [(args.file_a, "A"), (args.file_b, "B"), (args.file_o, "O")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {name} not found: {path}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}\n"
                                f"Please upload mergebert.pt to {args.checkpoint}")

    # Read file contents
    with open(args.file_a, 'r', encoding='utf-8') as f:
        a_text = f.read()
    with open(args.file_b, 'r', encoding='utf-8') as f:
        b_text = f.read()
    with open(args.file_o, 'r', encoding='utf-8') as f:
        o_text = f.read()

    # Resolve conflict
    resolved_text = resolve_conflict(a_text, b_text, o_text,
                                     args.model_path, args.checkpoint)

    print("\n" + "="*50)
    print("--- Resolved Code ---")
    print("="*50)
    print(resolved_text)

    # Optionally save to file
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(resolved_text)
        print(f"\n✓ Resolved code saved to: {args.output}")


def is_notebook():
    """检测是否在 Jupyter/Colab notebook 环境中运行"""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except:
        pass
    return False


if __name__ == "__main__":
    # 在 notebook 环境中不自动运行 main()
    # 用户应该使用 resolve_conflict() 函数
    if not is_notebook():
        main()
    else:
        print("="*60)
        print("在 Colab/Jupyter Notebook 中运行推理，请使用以下方式：")
        print("="*60)
        print("""
# 方式 1: 直接传入代码字符串
from inferencecolab import resolve_conflict

a_code = '''你的版本代码'''
b_code = '''他们的版本代码'''
o_code = '''基础版本代码'''

result = resolve_conflict(a_code, b_code, o_code)
print(result)

# 方式 2: 从文件读取
with open('file_a.txt') as f: a_code = f.read()
with open('file_b.txt') as f: b_code = f.read()
with open('file_o.txt') as f: o_code = f.read()

result = resolve_conflict(a_code, b_code, o_code)
print(result)
""")

