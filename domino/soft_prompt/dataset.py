import json
import torch
from torch.utils.data import Dataset
from typing import Dict

class SoftDataset(Dataset):
    """Dataset class for soft prompt tuning."""
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        print(f"Loading data from {data_path}...")
        self.raw_data = self.load_data(data_path)

    def __len__(self):
        return len(self.raw_data)

    def load_data(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.raw_data[idx]
        
        text = sample["question_content"]

        encoding = self.tokenizer(text, truncation=True, padding=False, max_length=self.max_seq_len, return_tensors="pt")
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
