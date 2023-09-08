from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass
class DataCollatorForSLiC:
    tokenizer: Callable
    model: Optional[Any] = None
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        features = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors=return_tensors,
        )
        features = {k:v.flatten(start_dim=0, end_dim=1) for k,v in features.items()}
        # prepare decoder_input_ids
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
        features["decoder_input_ids"] = decoder_input_ids
        
        return features