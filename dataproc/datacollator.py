from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NewType, Optional, Set, Tuple, Union

import numpy as np
import tokenizers
import torch
from transformers.data.data_collator import (
    DataCollatorWithPadding,
    InputDataClass,
    torch_default_data_collator,
)

def mask_context_torch_data_collator(
    features: List[InputDataClass], 
    rep_start_tokenid: int, 
    additional_mask_tokens: Set[int] = set(),
) -> Dict[str, Any]:
    pad_value = -100
    batch = torch_default_data_collator(features)
    for i in range(len(batch["labels"])):
        orig_labels = batch["labels"][i]
        pad_mask = torch.zeros_like(orig_labels).to(bool)
        for idx, label in enumerate(orig_labels):
            if label.item() in additional_mask_tokens:
                pad_mask[idx] = True
            if label.item() == rep_start_tokenid:
                pad_mask[:idx+1] = True
                
        batch["labels"][i][pad_mask] = pad_value
    return batch

def replace_with_unknown_agent_torch_data_collator(
    features: List[InputDataClass], 
    replace_unknown_prob: float = 0.,
    unknown_agent_id: Optional[int] = None, 
    num_agent_ids: int = 0,
) -> Dict[str, Any]:
    #print("features:", features)
    batch = torch_default_data_collator(features)

    if replace_unknown_prob > 0:
        for i in range(len(batch['input_ids'])):
            if torch.rand(1).item() < replace_unknown_prob:
                agent_id_mask = (batch['input_ids'][i] > unknown_agent_id) & (
                    batch['input_ids'][i] < unknown_agent_id + num_agent_ids)
                batch['input_ids'][i][agent_id_mask] = unknown_agent_id
                # in labels is different from input_ids
                agent_id_mask = (batch['labels'][i] > unknown_agent_id) & (
                    batch['labels'][i] < unknown_agent_id + num_agent_ids)
                batch['labels'][i][agent_id_mask] = unknown_agent_id

    return batch