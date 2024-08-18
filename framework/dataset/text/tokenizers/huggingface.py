
import os
from typing import List, Union, Dict, Any, Union, Iterator


class HuggingfaceVocabulary:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.tokenizer)

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]):
        pass

    def indices_to_sentence(self, indices: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(indices)

    def sentence_to_indices(self, sentence: str) -> List[int]:
        return self.tokenizer(sentence)["input_ids"]

    def __call__(self, seq: Union[List[Union[str, int]], str]) -> List[Union[int, str]]:
        if seq is None or (isinstance(seq, list) and not seq):
            return seq

        if isinstance(seq, str) or isinstance(seq[0], str):
            return self.sentence_to_indices(seq)
        else:
            return self.indices_to_sentence(seq)

    def to_string(self, seq: List[int]) -> str:
        return self.tokenizer.decode(indices)
