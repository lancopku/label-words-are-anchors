import warnings
from copy import deepcopy

import torch

from ..utils.data_wrapper import format_s_dict
from ..utils.other import TensorStrFinder


class ContextSolver:
    def __init__(self, task_name, tokenizer=None):
        assert task_name in ['sst2', 'trec', 'agnews', 'emo']
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.format_s = format_s_dict[task_name]
        self.parse_format_s()

    def parse_format_s(self):
        self.X_prefix = self.format_s.split('\n')[0].split(':')[0] + ':'
        self.Y_prefix = self.format_s.split('\n')[1].split(':')[0] + ':'

    def get_empty_demo_context(self, context: str, only_demo_part=True):
        context = context.split('\n')
        for i, line in enumerate(context[:-2]):
            if self.X_prefix in line:
                line = self.X_prefix
            elif self.Y_prefix in line:
                line = line
            else:
                raise warnings.warn('Global prefix or other str exists!')
            context[i] = line
        if only_demo_part:
            context = context[:-2]
        context = '\n'.join(context)
        return context

    def get_mask_strings_and_match_before(self, context, input_ids, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        poss = torch.where(input_ids == tokenizer.encode('\n', add_special_tokens=False)[0])[0]
        if len(poss) >= 2:
            match_before = poss[-2] + 1
        else:
            match_before = None

        list_s = []
        list_s.append(self.X_prefix)
        list_s.append('\n' + self.X_prefix)
        context = context.split('\n')
        for i, line in enumerate(context[:-2]):
            if self.X_prefix in line:
                pass
            elif self.Y_prefix in line:
                list_s.append('\n' + line)
                list_s.append('\n' + line + '\n')
            else:
                raise warnings.warn('Global prefix or other str exists!')
        return list_s, match_before

    def get_mask(self, input_ids, tokenizer=None):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        if len(input_ids.shape) == 2:
            assert input_ids.shape[0] == 1
            input_ids = input_ids[0]
        if tokenizer is None:
            tokenizer = self.tokenizer
        context = tokenizer.decode(input_ids)
        list_s, match_before = self.get_mask_strings_and_match_before(context, input_ids=input_ids,
                                                                      tokenizer=tokenizer)
        tensor_str_finder = TensorStrFinder(tokenizer=tokenizer)
        mask = tensor_str_finder.get_strs_mask_in_tensor(list_s=list_s, t=input_ids,
                                                         match_before=match_before)
        return mask
