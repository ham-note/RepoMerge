import datetime
import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import logging
from typing import Callable, List # Added for type hints in utility functions

# ==================== Colab 路径配置 ====================
# 数据目录 (上传数据文件到此目录)
DATA_DIR = "/content/"
# 使用 Hugging Face 在线模型
MODEL_PATH = "huggingface/CodeBERTa-small-v1"
# 模型保存目录
SAVED_MODEL_DIR = "/content/saved_models/"

# ==================== 训练超参数 ====================
NUM_LABELS = 9
EPOCHS = 10
BATCH_SIZE = 8  # Reduced from 32 to 8 to mitigate CUDA out of memory errors
LEARNING_RATE = 5e-5
MAX_LENGTH = 512 # Max sequence length for tokenizer, added from inference script

# ==================== 标签映射 ====================
LABEL_MAP = {
    "select_a": 0,
    "select_b": 1,
    "select_o": 2,
    "concat_ab": 3,
    "concat_ba": 4,
    "select_a_del_o": 5,
    "select_b_del_o": 6,
    "concat_ab_del_o": 7,
    "concat_ba_del_o": 8,
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ==================== 模型定义 (内嵌 model.py) ====================
class MergeBERTModel(nn.Module):
    """MergeBERT model for merge conflict resolution."""
    def __init__(self, encoder_path, num_labels=9):
        super(MergeBERTModel, self).__init__()
        self.num_labels = num_labels

        # The paper describes 4 parallel encoders. We'll instantiate one and use it four times.
        # This enforces weight sharing across the encoders, which is a common practice.
        config = RobertaConfig.from_pretrained(encoder_path)
        # Load the model in full-precision (float32) by default. AMP will handle mixed precision.
        self.encoder = RobertaModel.from_pretrained(encoder_path, config=config)

        # Aggregation weights (learnable) - default to float32
        self.aggregation_weights = nn.Parameter(torch.ones(4))

        # Classification head - default to float32
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, a_o_ids, o_a_ids, b_o_ids, o_b_ids, labels=None):
        """
        Forward pass of the model.
        For the MVP, we are not using the edit type embeddings yet.
        `*_ids` are tensors of shape (batch_size, seq_length).
        """
        # Pass each input through the shared encoder
        output_ao = self.encoder(input_ids=a_o_ids).pooler_output
        output_oa = self.encoder(input_ids=o_a_ids).pooler_output
        output_bo = self.encoder(input_ids=b_o_ids).pooler_output
        output_ob = self.encoder(input_ids=o_b_ids).pooler_output

        # Weighted aggregation of the [CLS] token representations (pooler_output)
        agg_weights = nn.functional.softmax(self.aggregation_weights, dim=0)

        aggregated_output = (
            agg_weights[0] * output_ao +
            agg_weights[1] * output_oa +
            agg_weights[2] * output_bo +
            agg_weights[3] * output_ob
        )

        # Classification
        logits = self.classifier(aggregated_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits
        }


# ==================== Utility functions for alignment (copied from inferencecolab.py) ====================
edit_seq_tokens = {
    "eql": "=", "add": "+", "del": "-", "rep": "↔", "padding": "[PAD]"
}

class OffsetRange:
    def __init__(self, start: int, end: int):
        self.start, self.end = start, end
    def __len__(self):
        return self.end - self.start

class SequenceDiff:
    def __init__(self, seq1Range: OffsetRange, seq2Range: OffsetRange):
        self.seq1Range, self.seq2Range = seq1Range, seq2Range

def compute(sequence1: List[str], sequence2: List[str], equalityScore: Callable[[str, str], float] = lambda x, y: 1 if x == y else 0) -> List[SequenceDiff]:
    if not sequence1 or not sequence2: return [SequenceDiff(OffsetRange(0, len(sequence1)), OffsetRange(0, len(sequence2)))]
    lcsLengths, directions, lengths = np.zeros((len(sequence1), len(sequence2))), np.zeros((len(sequence1), len(sequence2))), np.zeros((len(sequence1), len(sequence2)))
    for s1 in range(len(sequence1)): # type: ignore
        for s2 in range(len(sequence2)): # type: ignore
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

def get_edit_sequence(before: List[str], after: List[str]) -> tuple[List[str], List[str], List[str]]:
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


# ==================== 数据集定义 ====================
class MergeDataset(Dataset):
    """Custom PyTorch Dataset for the merge conflict data."""
    def __init__(self, file_path, tokenizer, max_length=MAX_LENGTH, limit=None):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        if limit:
            self.samples = self.samples[:limit]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def _pad_and_truncate(self, token_ids: List[int]) -> List[int]:
        if len(token_ids) > self.max_length:
            return token_ids[:self.max_length]
        else:
            return token_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(token_ids))

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Expecting raw text in the JSON for 'a', 'b', 'o'
        a_text = sample.get('a_text', '')
        b_text = sample.get('b_text', '')
        o_text = sample.get('o_text', '')

        # Tokenize raw texts using the CodeBERTa tokenizer
        a_tokens = self.tokenizer.tokenize(a_text)
        b_tokens = self.tokenizer.tokenize(b_text)
        o_tokens = self.tokenizer.tokenize(o_text)

        # Generate four edit sequences (list of string tokens) as per MergeBERT paper:
        # E_AO: Edit operations from O to A
        # E_OA: Edit operations from A to O
        # E_BO: Edit operations from O to B
        # E_OB: Edit operations from B to O
        ao_edit_seq, _, _ = get_edit_sequence(o_tokens, a_tokens) # O to A
        oa_edit_seq, _, _ = get_edit_sequence(a_tokens, o_tokens) # A to O
        bo_edit_seq, _, _ = get_edit_sequence(o_tokens, b_tokens) # O to B
        ob_edit_seq, _, _ = get_edit_sequence(b_tokens, o_tokens) # B to O

        # Convert edit sequence tokens (strings like '=', '+') to IDs and pad
        # Note: We assume these special edit tokens are recognized by the tokenizer
        # or converted to UNK which is handled during training.
        a_o_ids = self._pad_and_truncate(self.tokenizer.convert_tokens_to_ids(ao_edit_seq))
        o_a_ids = self._pad_and_truncate(self.tokenizer.convert_tokens_to_ids(oa_edit_seq))
        b_o_ids = self._pad_and_truncate(self.tokenizer.convert_tokens_to_ids(bo_edit_seq))
        o_b_ids = self._pad_and_truncate(self.tokenizer.convert_tokens_to_ids(ob_edit_seq))

        return {
            "a_o_ids": torch.tensor(a_o_ids, dtype=torch.long),
            "o_a_ids": torch.tensor(o_a_ids, dtype=torch.long),
            "b_o_ids": torch.tensor(b_o_ids, dtype=torch.long),
            "o_b_ids": torch.tensor(o_b_ids, dtype=torch.long),
            "labels": torch.tensor(sample['numeric_label'], dtype=torch.long),
            "descriptive_label": sample['descriptive_label']
        }


# ==================== 评估指标 ====================
def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# ==================== 训练函数 ====================
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler() # Initialize GradScaler for mixed precision

    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()

        a_o_ids = batch['a_o_ids'].to(device)
        o_a_ids = batch['o_a_ids'].to(device)
        b_o_ids = batch['b_o_ids'].to(device)
        o_b_ids = batch['o_b_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast(): # Enable autocast for mixed precision
            outputs = model(a_o_ids, o_a_ids, b_o_ids, o_b_ids, labels=labels)
            loss = outputs['loss']
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                loss = loss.mean()

        scaler.scale(loss).backward() # Scale loss and perform backward pass
        scaler.step(optimizer) # Update optimizer
        scaler.update() # Update scaler for next iteration
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


# ==================== 评估函数 ====================
def eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_numeric_labels = []
    all_descriptive_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            a_o_ids = batch['a_o_ids'].to(device)
            o_a_ids = batch['o_a_ids'].to(device)
            b_o_ids = batch['b_o_ids'].to(device)
            o_b_ids = batch['o_b_ids'].to(device)
            labels = batch['labels'].to(device)
            descriptive_labels = batch['descriptive_label']

            with torch.cuda.amp.autocast(): # Enable autocast for mixed precision during evaluation
                outputs = model(a_o_ids, o_a_ids, b_o_ids, o_b_ids, labels=labels)
                loss = outputs['loss']
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                loss = loss.mean()
            logits = outputs['logits']

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_numeric_labels.extend(labels.cpu().numpy())
            all_descriptive_labels.extend(descriptive_labels)

    avg_loss = total_loss / len(data_loader)

    # --- Detailed Analysis ---
    results_by_label = defaultdict(lambda: {"true": [], "pred": []})
    for true_label, pred_label, desc_label in zip(all_numeric_labels, all_preds, all_descriptive_labels):
        results_by_label[desc_label]["true"].append(true_label)
        results_by_label[desc_label]["pred"].append(pred_label)

    logging.info("\n" + "="*20 + " DETAILED EVALUATION PER CATEGORY " + "="*20 + "\n")

    for desc_label, data in sorted(results_by_label.items()):
        y_true = data["true"]
        y_pred = data["pred"]

        if not y_true:
            continue

        logging.info(f"--- Classification Report for Category: '{desc_label}' (count: {len(y_true)}) ---")

        all_numeric_labels_in_group = sorted(list(set(y_true) | set(y_pred)))
        target_names = [REVERSE_LABEL_MAP.get(l, f"unknown_label_{l}") for l in all_numeric_labels_in_group]

        try:
            report = classification_report(
                y_true,
                y_pred,
                labels=all_numeric_labels_in_group,
                target_names=target_names,
                zero_division=0
            )
            logging.info("\n" + report)
        except Exception as e:
            logging.warning(f"Could not generate classification report for '{desc_label}': {e}")
            acc = accuracy_score(y_true, y_pred)
            logging.info(f"  - Overall Accuracy for this group: {acc:.4f}")

        logging.info("--------------------------------------------------------------------\n")

    # Overall metrics
    metrics = compute_metrics(all_preds, all_numeric_labels)
    return avg_loss, metrics


# ==================== 主函数 ====================
def main():
    """Main function to run the training."""
    # 创建保存模型的目录
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

    log_filename = f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info("\n\n" + "="*50 + " NEW TRAINING RUN " + "="*50 + "\n")

    # 检测设备 - Colab 优先使用 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Initialize tokenizer once for the entire training run
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    logging.info(f"Loaded tokenizer from: {MODEL_PATH}")

    # 数据集名称 (与预处理输出的文件名匹配)
    dataset_name = "data"
    train_file = os.path.join(DATA_DIR, f"{dataset_name}_train.json")
    val_file = os.path.join(DATA_DIR, f"{dataset_name}_validation.json")

    # 检查数据文件是否存在
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练数据文件不存在: {train_file}\n请上传 {dataset_name}_train.json 到 {DATA_DIR}. 文件应包含 'a_text', 'b_text', 'o_text', 'numeric_label', 'descriptive_label' 字段。")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"验证数据文件不存在: {val_file}\n请上传 {dataset_name}_validation.json 到 {DATA_DIR}. 文件应包含 'a_text', 'b_text', 'o_text', 'numeric_label', 'descriptive_label' 字段。")

    train_dataset = MergeDataset(train_file, tokenizer, max_length=MAX_LENGTH)
    val_dataset = MergeDataset(val_file, tokenizer, max_length=MAX_LENGTH)

    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    model = MergeBERTModel(encoder_path=MODEL_PATH, num_labels=NUM_LABELS)
    # Clear CUDA cache before moving the model to GPU
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model.to(device)

    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_accuracy = 0

    for epoch in range(EPOCHS):
        logging.info(f"\nEpoch {epoch + 1}/{EPOCHS}")
        logging.info("-" * 30)

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logging.info(f"Train loss: {train_loss:.4f}")

        val_loss, metrics = eval_epoch(model, val_loader, device)
        logging.info(f"Validation loss: {val_loss:.4f}")
        logging.info(f"Overall validation metrics: {metrics}")

        if metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = metrics['accuracy']
            model_to_save = model.module if hasattr(model, 'module') else model
            save_path = os.path.join(SAVED_MODEL_DIR, 'mergebert.pt')
            torch.save(model_to_save.state_dict(), save_path)
            logging.info(f"\u2713 Best model saved to {save_path}")

    logging.info("\n" + "="*50)
    logging.info(f"Training completed! Best validation accuracy: {best_val_accuracy:.4f}")
    logging.info(f"Model saved to: {SAVED_MODEL_DIR}mergebert.pt")
    logging.info("="*50)


if __name__ == "__main__":
    main()
