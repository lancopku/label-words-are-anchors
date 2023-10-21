import warnings
from typing import Callable, Optional, List, Union
from functools import wraps, partial

import numpy as np
import torch
from datasets import concatenate_datasets
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, DataCollatorWithPadding, Trainer
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from ..util_classes.predictor_classes import Predictor
from ..util_classes.setter_with_restore import SetterWithRestore
from ..lm_apis.lm_api_base import LMForwardAPI
from ..utils.data_wrapper import wrap_dataset, tokenize_dataset, remove_str_columns
from ..utils.other import sample_two_set_with_shot_per_class


class QKVGetter:
    def __init__(self):
        self.q = None
        self.k = None
        self.v = None

    def register_qkv(self, q, k, v):
        self.q = q
        self.k = k
        self.v = v


def wrap_fn_save_query_key_value(qkvgetter: QKVGetter):
    def decorator(func):
        def wrapper(query, key, value, attention_mask=None, head_mask=None):
            qkvgetter.register_qkv(query, key, value)
            return func(query, key, value, attention_mask=attention_mask, head_mask=head_mask)

        return wrapper

    return decorator


class QKVGetterManger:
    def __init__(self, model: LMForwardAPI, predictor: Predictor):
        self.model = model
        self.qkv = None
        self.setter_with_restore = SetterWithRestore()
        self.qkvgetters = self.register_qkvgetter_to_model()
        self.setter_with_restore.set(self.model.forward,
                                     manager_decoractor(self)(self.model.forward))
        self.predictor = predictor

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids):
        self._input_ids = input_ids

    def register_input_ids(self, input_ids):
        self.input_ids = input_ids

    def register_qkvgetter_to_model(self):
        qkvgetters = []
        if self.model.model_name in ['gpt2-xl','gpt-j-6b']:
            for i, layer in enumerate(self.model.model.transformer.h):
                qkvgetter = QKVGetter()
                qkvgetters.append(qkvgetter)
                wrapped_attn = wrap_fn_save_query_key_value(qkvgetter=qkvgetter)(
                    layer.attn._attn)
                self.setter_with_restore.set(layer.attn._attn, wrapped_attn)
        else:
            raise NotImplementedError(f'{self.model.model_name} not supported yet')
        return qkvgetters

    def unregister(self):
        self.setter_with_restore.restore_all()

    def collect_qkv(self):
        qkv = []
        for qkvgetter in self.qkvgetters:
            qkv.append((qkvgetter.q, qkvgetter.k, qkvgetter.v))
        inputs = {'input_ids': self.input_ids}
        qkv = self.predictor._cal_all_key_and_values_of_class(inputs=inputs, past_key_values=qkv,
                                                              one_class_one_list=True,
                                                              include_final=True)
        return qkv


def manager_decoractor(manager: QKVGetterManger):
    def model_forward_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            input_ids = kwargs.get('input_ids', None)
            if input_ids is None:
                input_ids = args[0]
            manager.register_input_ids(input_ids)
            results = fn(*args, **kwargs)
            assert 'qkv' not in results
            results['qkv'] = manager.collect_qkv()
            return results

        return wrapper

    return model_forward_decorator


def prepare_analysis_dataset(demonstration, args, seed, dataset, tokenizer):
    if args.sample_from == 'test':
        if len(dataset['test']) < args.actual_sample_size:
            args.actual_sample_size = len(dataset['test'])
            warnings.warn(
                f"sample_size: {args.sample_size} is larger than test set size: {len(dataset['test'])},"
                f"actual_sample_size is {args.actual_sample_size}")
        test_sample = dataset['test'].shuffle(seed=seed).select(range(args.actual_sample_size))
        demo_dataset = wrap_dataset(test_sample, demonstration, args.label_dict,
                                    args.task_name)
        demo_dataset = tokenize_dataset(demo_dataset, tokenizer)
    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")

    return demo_dataset

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=axis, keepdims=True)


def cal_results(demonstraions, model, tokenizer, training_args, args, seed, dataset):
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, pad_to_multiple_of=1, max_length=1024)
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator)
    demo_dataset, empty_test_dataset = prepare_analysis_dataset(demonstraions, args, seed, dataset,
                                                                tokenizer)
    demo_dataset = remove_str_columns(demo_dataset)
    demo_y = trainer.predict(demo_dataset, ignore_keys=['results'])
    labels = np.array(demo_dataset['label'])
    pred_label = demo_y.predictions[0].argmax(-1)
    return labels, pred_label

