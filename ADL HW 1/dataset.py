from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        return {
            'text': [seq['text'] for seq in samples],
            'intent': [(seq['intent'] if 'intent' in seq else []) for seq in samples],
            'id': [seq['id'] for seq in samples],
            'padding': [pad_to_len([self.vocab.encode((seq['text'].split()))], self.max_len, 0) for seq in samples],
            'label': [(self.label2idx(seq['intent']) if 'intent' in seq else []) for seq in samples]
        }

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class TagClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        tag_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.tag_mapping = tag_mapping
        self._idx2tag = {idx: intent for intent, idx in self.tag_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.tag_mapping)

    def collate_fn(self, samples) -> Dict:
        return {
            'pad_tokens': [pad_to_len([self.vocab.encode((seq['tokens']))], self.max_len, 0) for seq in samples],
            'pad_tags': [(([self.tag2idx(tags) for tags in seq['tags']])[:self.max_len] + [-1] * max(0, self.max_len - len(seq['tags'])) if 'tags' in seq else [])for seq in samples],
            'id': [seq['id'] for seq in samples],
        }

    def tag2idx(self, tag: str):
        return self.tag_mapping[tag]

    def idx2tag(self, idx: int):
        return self._idx2tag[idx]