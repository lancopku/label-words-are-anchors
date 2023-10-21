import pickle
import time
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import List
import os
from transformers.hf_argparser import HfArgumentParser
import torch
import torch.nn.functional as F

from ..lm_apis.lm_api_base import LMForwardAPI
from ..util_classes.context_solver import ContextSolver
from ..utils.data_wrapper import wrap_dataset, tokenize_dataset, wrap_dataset_with_instruct, \
    remove_str_columns
from ..utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from ..utils.random_utils import set_seed
from ..utils.other import load_args, set_gpu, sample_two_set_with_shot_per_class
from transformers import Trainer, TrainingArguments, PreTrainedModel, AutoModelForCausalLM, \
    AutoTokenizer, DataCollatorWithPadding
from ..utils.load_local import get_model_layer_num
from ..util_classes.arg_classes import CompressTimeArgs
from ..utils.prepare_model_and_tokenizer import load_model_and_tokenizer, get_label_id_dict_for_args

def clip_sample(sample,tokenizer):
    input_ids = torch.tensor(sample['input_ids'])
    attention_mask = torch.tensor(sample['attention_mask'])
    eos_token_id = tokenizer.eos_token_id
    actual_len = (input_ids!=eos_token_id).sum()
    input_ids = input_ids[:actual_len]
    attention_mask = attention_mask[:actual_len]
    sample['input_ids'] = input_ids.reshape(1,-1)
    sample['attention_mask'] = attention_mask.reshape(1,-1)
    return sample

def my_dict_to(d: dict, device):
    d['input_ids'] = torch.tensor(d['input_ids']).to(device)
    d['attention_mask'] = torch.tensor(d['attention_mask']).to(device)
    return d

def time_recorder(fn):
    def wrapper(*args,**kwargs):
        start = time.time()
        res = fn(*args,**kwargs)
        end = time.time()
        print(f"{fn.__name__} time: {end-start}")
        return res, end-start
    return wrapper

@time_recorder
def main(model,datas):
    with torch.no_grad():
        for data in datas:
            data = my_dict_to(data,model.device)
            _ = model(**data)

def compress_time(args: CompressTimeArgs):
    if os.path.exists(args.save_file_name):
        return
    set_gpu(args.gpu)
    if args.sample_from == 'test':
        dataset = load_huggingface_dataset_train_and_test(args.task_name)
    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")

    model, tokenizer = load_model_and_tokenizer(args)
    args.label_id_dict = get_label_id_dict_for_args(args, tokenizer)

    model = LMForwardAPI(model=model, model_name=args.model_name, tokenizer=tokenizer,
                         device='cuda:0',
                         label_dict=args.label_dict)

    num_layer = get_model_layer_num(model=model.model, model_name=args.model_name)

    context_solver = ContextSolver(task_name=args.task_name, tokenizer=tokenizer)

    def prepare_analysis_dataset(seed):
        demonstration, _ = sample_two_set_with_shot_per_class(dataset['train'],
                                                              args.demonstration_shot,
                                                              0, seed, label_name='label',
                                                              a_total_shot=args.demonstration_total_shot)
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

            context = demo_dataset[0]['sentence']
            instruct = context_solver.get_empty_demo_context(context, only_demo_part=True)

            empty_demo_dataset = wrap_dataset_with_instruct(test_sample, instruct, args.label_dict,
                                                            args.task_name)
            empty_demo_dataset = tokenize_dataset(empty_demo_dataset, tokenizer)

            no_demo_dataset = wrap_dataset(test_sample, [], args.label_dict,
                                           args.task_name)
            no_demo_dataset = tokenize_dataset(no_demo_dataset, tokenizer)
        else:
            raise NotImplementedError(f"sample_from: {args.sample_from}")

        return demo_dataset, empty_demo_dataset, no_demo_dataset


    ys = []
    for seed in args.seeds:
        analysis_dataset, analysis_empty_demo_dataset, analysis_no_demo_dataset = prepare_analysis_dataset(
            seed)

        demo_dataset = analysis_dataset.map(partial(clip_sample, tokenizer=tokenizer))
        no_demo_dataset = analysis_no_demo_dataset.map(partial(clip_sample, tokenizer=tokenizer))

        model.results_args = {'output_hidden_states': True,
                              'output_attentions': True, 'use_cache': True}
        data = demo_dataset[0]
        mask = context_solver.get_mask(data['input_ids'])
        data = my_dict_to(data, model.device)
        model.use_past_key_values = False
        with torch.no_grad():
            y = model(**data)
        past_key_values = y['results'].past_key_values
        past_key_values = tuple(
            tuple(t[:, :, mask, :] for t in tup) for tup in past_key_values)

        model.use_past_key_values = True
        model.past_key_values = past_key_values
        y1 = main(model, no_demo_dataset)

        model.use_past_key_values = False
        model.past_key_values = None
        y2 = main(model, demo_dataset)

        ys.append((y1, y2))

    os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
    with open(args.save_file_name, 'wb') as f:
        pickle.dump(ys, f)
