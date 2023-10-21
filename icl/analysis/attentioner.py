import random
import warnings
from typing import Callable, Optional, List, Union
from functools import wraps, partial
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class AttentionAdapterBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag = False

    def forward(self, attn_weights):
        if self.use_flag:
            return self._forward(attn_weights)
        else:
            return attn_weights

    def _forward(self, attn_weights):
        raise NotImplementedError

    def register_input_ids(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


def gpt2_attn(self, query, key, value, attention_mask=None, head_mask=None, attention_adapter=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    if not self.is_cross_attention:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights,
                                   self.masked_bias.to(attn_weights.dtype))

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    if attention_adapter is not None:
        attn_weights = attention_adapter(attn_weights)

    attn_weights = nn.Softmax(dim=-1)(attn_weights)

    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


def gptj_attn(self, query, key, value, attention_mask=None, head_mask=None, attention_adapter=None):
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]

    query = query.to(torch.float32)
    key = key.to(torch.float32)

    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    mask_value = torch.finfo(attn_weights.dtype).min
    mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    attn_weights = attn_weights / self.scale_attn

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    if attention_adapter is not None:
        attn_weights = attention_adapter(attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.to(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


class AttentionerManagerBase:
    def __init__(self, model: PreTrainedModel, attention_adapters: List[AttentionAdapterBase]):
        self.model = model
        self.attention_adapters = attention_adapters
        self.model.forward = manager_decoractor(self)(self.model.forward)
        self.register_attentioner_to_model()

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids):
        self._input_ids = input_ids
        for attention_adapter in self.attention_adapters:
            attention_adapter.register_input_ids(input_ids)

    def register_input_ids(self, input_ids):
        self.input_ids = input_ids

    def register_attentioner_to_model(self):
        raise NotImplementedError

    def set_attentioner_state(self, use_flag: bool,
                              attention_adapter_idxs: Optional[Union[int, List[int]]] = None):
        if attention_adapter_idxs is None:
            attention_adapter_idxs = range(len(self.attention_adapters))
        elif isinstance(attention_adapter_idxs, int):
            attention_adapter_idxs = [attention_adapter_idxs]
        for attention_adapter_idx in attention_adapter_idxs:
            self.attention_adapters[attention_adapter_idx].use_flag = use_flag


def manager_decoractor(manager: AttentionerManagerBase):
    def model_forward_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            input_ids = kwargs.get('input_ids', None)
            if input_ids is None:
                input_ids = args[0]
            manager.register_input_ids(input_ids)
            return fn(*args, **kwargs)

        return wrapper

    return model_forward_decorator


class GPT2AttentionerManager(AttentionerManagerBase):
    def __init__(self, model: PreTrainedModel, attention_adapters: List[AttentionAdapterBase]):
        super().__init__(model, attention_adapters)

    def register_attentioner_to_model(self):
        for i, layer in enumerate(self.model.transformer.h):
            layer.attn._attn = partial(gpt2_attn, layer.attn,
                                       attention_adapter=self.attention_adapters[i])


class GPTJAttentionerManager(AttentionerManagerBase):
    def __init__(self, model: PreTrainedModel, attention_adapters: List[AttentionAdapterBase]):
        super().__init__(model, attention_adapters)

    def register_attentioner_to_model(self):
        for i, layer in enumerate(self.model.transformer.h):
            layer.attn._attn = partial(gptj_attn, layer.attn,
                                       attention_adapter=self.attention_adapters[i])


class AttentionAdapter(AttentionAdapterBase):
    def __init__(self, label_id_dict, pad_token_id, task_name, tokenizer, window_size=None) -> None:
        super().__init__()
        self.label_id_dict = label_id_dict
        self.pad_token_id = pad_token_id
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.window_size = window_size
        if self.window_size is not None:
            warnings.warn(
                "window_size is not used")

        if task_name == 'sst2':
            self.prefix_idxs = [tokenizer.encode('Sentiment')[-1], tokenizer.encode(':')[0]]
        elif task_name == 'agnews':
            self.prefix_idxs = [tokenizer.encode('Answer')[-1], tokenizer.encode(':')[0]]
        elif task_name == 'trec':
            self.prefix_idxs = [tokenizer.encode(' Type')[-1], tokenizer.encode(':')[0]]
        elif task_name == 'emo':
            self.prefix_idxs = [tokenizer.encode('Emotion')[-1], tokenizer.encode(':')[0]]
        else:
            raise NotImplementedError(f"task_name: {task_name}")

    def get_pos(self, input_ids, label_id_dict, pad_token_id):
        ori_input_ids = input_ids.detach().clone()
        final_pos = torch.ne(ori_input_ids, pad_token_id).int().sum(-1) - 1
        device = ori_input_ids.device
        bsz, sql = ori_input_ids.shape
        class_poss = []
        for idx in label_id_dict.values():
            class_idx = idx
            for offset, prefix_idx in enumerate(reversed(self.prefix_idxs)):
                class_idx += prefix_idx * 100000 ** (offset + 1)
            input_ids = ori_input_ids.detach().clone()
            input_ids[:, 1:] += ori_input_ids[:, :-1] * 100000
            input_ids[:, 2:] += ori_input_ids[:, :-2] * 100000 * 100000
            class_pos = torch.arange(sql, device=device).unsqueeze(0).repeat(bsz, 1)[
                input_ids == class_idx].reshape(bsz, -1)
            class_poss.append(class_pos)
        return class_poss, final_pos

    def _forward(self, attn_weights):
        class_poss, final_pos = self.get_pos(self.input_ids, self.label_id_dict, self.pad_token_id)
        bsz, sql = self.input_ids.shape
        for class_pos in class_poss:
            for b_idx in range(bsz):
                for single_class_pos in class_pos[b_idx]:
                    attn_weights[b_idx, :,
                    single_class_pos: single_class_pos + self.window_size + 1,
                    :single_class_pos] = -10000.
        return attn_weights

class AttentionAdapterNonLabel(AttentionAdapterBase):
    def __init__(self, label_id_dict, pad_token_id, task_name, tokenizer, window_size=None) -> None:
        super().__init__()
        self.label_id_dict = label_id_dict
        self.pad_token_id = pad_token_id
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.window_size = window_size
        if self.window_size is not None:
            warnings.warn(
                "window_size is not used")

        if task_name == 'sst2':
            self.prefix_idxs = [tokenizer.encode('Sentiment')[-1], tokenizer.encode(':')[0]]
        elif task_name == 'agnews':
            self.prefix_idxs = [tokenizer.encode('Answer')[-1], tokenizer.encode(':')[0]]
        elif task_name == 'trec':
            self.prefix_idxs = [tokenizer.encode(' Type')[-1], tokenizer.encode(':')[0]]
        elif task_name == 'emo':
            self.prefix_idxs = [tokenizer.encode('Emotion')[-1], tokenizer.encode(':')[0]]
        else:
            raise NotImplementedError(f"task_name: {task_name}")

    def get_pos(self, input_ids, label_id_dict, pad_token_id):
        ori_input_ids = input_ids.detach().clone()
        final_pos = torch.ne(ori_input_ids, pad_token_id).int().sum(-1) - 1
        device = ori_input_ids.device
        bsz, sql = ori_input_ids.shape
        class_poss = []
        for idx in label_id_dict.values():
            class_idx = idx
            for offset, prefix_idx in enumerate(reversed(self.prefix_idxs)):
                class_idx += prefix_idx * 100000 ** (offset + 1)
            input_ids = ori_input_ids.detach().clone()
            input_ids[:, 1:] += ori_input_ids[:, :-1] * 100000
            input_ids[:, 2:] += ori_input_ids[:, :-2] * 100000 * 100000
            class_pos = torch.arange(sql, device=device).unsqueeze(0).repeat(bsz, 1)[
                input_ids == class_idx].reshape(bsz, -1)
            class_poss.append(class_pos)
        return class_poss, final_pos

    def _forward(self, attn_weights):
        class_poss, final_pos = self.get_pos(self.input_ids, self.label_id_dict, self.pad_token_id)
        # device = self.input_ids.device
        bsz, sql = self.input_ids.shape
        assert bsz == 1
        class_poss_flatten = torch.flatten(torch.cat(class_poss, dim=-1))
        non_label_idxs = set(range(final_pos[0] - 1)) - set(class_poss_flatten.tolist())
        random_word_idxs = random.sample(non_label_idxs, len(class_poss_flatten))

        for b_idx in range(bsz):
            for random_word_idx in random_word_idxs:
                attn_weights[b_idx, :,
                random_word_idx: random_word_idx + self.window_size + 1,
                :random_word_idx] = -10000.
        return attn_weights