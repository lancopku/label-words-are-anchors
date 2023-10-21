import functools
import warnings
from typing import Union, List, Optional

import numpy as np
import torch
from transformers import HfArgumentParser
import os
from .random_utils import np_temp_random

REDUCE_FN_MAPPINGS = {
    'sum': torch.sum,
    'mean': torch.mean,
    'none': lambda x: x
}


def apply_on_element(l, fn=None):
    if isinstance(l, torch.Tensor):
        l = l.tolist()
    if isinstance(l, list):
        return [apply_on_element(_, fn) for _ in l]
    elif isinstance(l, dict):
        return {k: apply_on_element(v, fn) for k, v in l.items()}
    else:
        return fn(l)


def show_words(logits, tokenizer, topk=5):
    token_ids = logits.topk(topk)[1]
    words = apply_on_element(token_ids, tokenizer.convert_ids_to_tokens)
    print(words)


def load_args(args_type, is_ipynb=False):
    if not is_ipynb:
        parser = HfArgumentParser((args_type,))
        args, = parser.parse_args_into_dataclasses()
    else:
        args = args_type()
    return args



def sample_two_set_with_shot_per_class(ori_data, a_shot, b_shot, seed, label_name: str = 'labels',
                                       a_total_shot=None, b_total_shot=None):
    a_label_count = {}
    b_label_count = {}
    a_data_idx = []
    b_data_idx = []
    all_indices = [_ for _ in range(len(ori_data))]
    np_temp_random(seed=seed)(np.random.shuffle)(all_indices)

    a_total_cnt = 0
    b_total_cnt = 0
    for index in all_indices:
        label = ori_data[index][label_name]
        if label < 0:
            continue

        if label not in a_label_count.keys():
            a_label_count[label] = 0
        if label not in b_label_count.keys():
            b_label_count[label] = 0

        if a_label_count[label] < a_shot:
            a_data_idx.append(index)
            a_label_count[label] += 1
            a_total_cnt += 1
        elif b_label_count[label] < b_shot:
            b_data_idx.append(index)
            b_label_count[label] += 1
            b_total_cnt += 1

        a_cond = a_total_shot is not None and a_total_cnt >= a_total_shot
        b_cond = (b_total_shot is not None and b_total_cnt >= b_total_shot) or (b_shot == 0)
        if a_cond and b_cond:
            warnings.warn(f"sampled {a_total_shot} and {b_total_shot} samples, ")

    a_data = ori_data.select(a_data_idx)
    b_data = ori_data.select(b_data_idx)
    return a_data, b_data


def dict_to(d: dict, device):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device)
    return d


def set_gpu(gpu_id: Union[str, int]):
    if isinstance(gpu_id, int):
        gpu_id = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


class TensorStrFinder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def find_tensor_in_tensor(self, a_tensor: Union[torch.Tensor, list], b_tensor: torch.Tensor,
                              return_mask=True, match_before: Optional[int] = None):
        if len(b_tensor.shape) == 2:
            assert b_tensor.shape[0] == 1
            b_tensor = b_tensor[0]
        if isinstance(a_tensor, list):
            a_tensor = torch.tensor(a_tensor)
        if a_tensor.device != b_tensor.device:
            a_tensor = a_tensor.to(b_tensor.device)

        window_size = len(a_tensor)
        b_windows = b_tensor.unfold(0, window_size, 1)

        matches = torch.all(b_windows == a_tensor, dim=1)

        positions = torch.nonzero(matches, as_tuple=True)[0]

        if return_mask:
            mask = torch.zeros_like(b_tensor, dtype=torch.bool)
            for pos in positions:
                if match_before is None or pos + window_size <= match_before:
                    mask[pos:pos + window_size] = True
            return mask

        return positions

    def find_str_in_tensor(self, s: str, t: torch.Tensor, return_mask=True, match_before=None):
        s_tokens = self.tokenizer.encode(s, add_special_tokens=False)
        s_tensor = torch.LongTensor(s_tokens)
        return self.find_tensor_in_tensor(s_tensor, t, return_mask=return_mask,
                                          match_before=match_before)

    def get_strs_mask_in_tensor(self, list_s: List[str], t: torch.Tensor, match_before=None):
        list_s_tokens = [self.tokenizer.encode(s, add_special_tokens=False) for s in list_s]
        list_s_tensor = [torch.LongTensor(s_tokens) for s_tokens in list_s_tokens]
        mask_tensor_list = [
            self.find_tensor_in_tensor(s_tensor, t, return_mask=True, match_before=match_before) for
            s_tensor in list_s_tensor]
        mask_tensor = functools.reduce(torch.logical_or, mask_tensor_list)
        return mask_tensor
