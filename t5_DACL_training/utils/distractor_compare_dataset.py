from typing import Callable
from transformers import T5Tokenizer
from torch.utils.data import Dataset
import torch

class DistractorComparelDataset(Dataset):
    
    def __init__(self, datas, tokenizer: Callable, sample_size = 4) -> None:
        super().__init__()
        self.sample_size = sample_size
        sentence = []
        sorted_distractors = []
        #answer = []
        for data in datas:
            if 'sentence' not in data or 'ranked_distractors' not in data:
                continue
            if len(data['ranked_distractors']) < self.sample_size:
                continue
            sentence.extend([data['sentence']] * self.sample_size)
            sorted_distractors.extend(data['ranked_distractors'][:self.sample_size])
            # sorted_distractors.extend(data['distractors_ranked'])
            # sorted_distractors.append(data['answer'])
            # answer.append(answer)
        
        self.sentence = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
        # print(sorted_distractors)
        self.sorted_distractors = tokenizer(sorted_distractors, padding=True, truncation=True, return_tensors="pt")
        #self.answer = tokenizer(answer, padding=True, truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.sentence['input_ids']) // self.sample_size

    def __getitem__(self, idx):
        return {'input_ids': [self.sentence['input_ids'][i] for i in range(self.sample_size * idx, self.sample_size * (idx + 1))],
                'attention_mask': [self.sentence['attention_mask'][i] for i in range(self.sample_size * idx, self.sample_size * (idx + 1))],
                'labels' : [self.sorted_distractors['input_ids'][i] for i in range(self.sample_size * idx, self.sample_size * (idx + 1))],
                'labels_attention_mask' : [self.sorted_distractors['attention_mask'][i] for i in range(self.sample_size * idx, self.sample_size * (idx + 1))]}