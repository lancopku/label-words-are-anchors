import pickle
import warnings
from dataclasses import dataclass, field
from typing import List
import os
from transformers.hf_argparser import HfArgumentParser
import torch
import torch.nn.functional as F

from ..lm_apis.lm_api_base import LMForwardAPI
from ..utils.data_wrapper import wrap_dataset, tokenize_dataset
from ..utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from ..utils.random_utils import set_seed
from ..utils.other import load_args, set_gpu, sample_two_set_with_shot_per_class
from transformers import Trainer, TrainingArguments, PreTrainedModel, AutoModelForCausalLM, \
    AutoTokenizer
from ..utils.load_local import convert_path_old, load_local_model_or_tokenizer, get_model_layer_num
from ..util_classes.arg_classes import NClassificationArgs
from ..utils.prepare_model_and_tokenizer import load_model_and_tokenizer, get_label_id_dict_for_args
from ..util_classes.predictor_classes import Predictor



def n_classify(args: NClassificationArgs):
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

    training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                      per_gpu_eval_batch_size=args.batch_size,
                                      per_gpu_train_batch_size=args.batch_size)

    num_layer = get_model_layer_num(model=model.model, model_name=args.model_name)

    def prepare_analysis_dataset(seed):
        demonstration, _ = sample_two_set_with_shot_per_class(dataset['train'],
                                                                          args.demonstration_shot,
                                                                          0,
                                                                          seed,
                                                                          label_name='label',
                                                                          a_total_shot=args.demonstration_total_shot)
        demonstration = demonstration.shuffle()
        if args.sample_from == 'test':
            if len(dataset['test']) < args.actual_sample_size:
                args.actual_sample_size = len(dataset['test'])
                warnings.warn(
                    f"sample_size: {args.sample_size} is larger than test set size: {len(dataset['test'])},"
                    f"actual_sample_size is {args.actual_sample_size}")
            test_sample = dataset['test'].shuffle(seed=seed).select(range(args.actual_sample_size))
            analysis_dataset = wrap_dataset(test_sample, demonstration, args.label_dict,
                                            args.task_name)
            analysis_dataset = tokenize_dataset(analysis_dataset, tokenizer)


        else:
            raise NotImplementedError(f"sample_from: {args.sample_from}")

        return analysis_dataset, demonstration

    ys = []
    for seed in args.seeds:
        analysis_dataset, demonstration = prepare_analysis_dataset(seed)

        trainer = Trainer(model=model, args=training_args)
        y = trainer.predict(analysis_dataset, ignore_keys=['results'])

        ys.append((y,))

    os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
    with open(args.save_file_name, 'wb') as f:
        pickle.dump([ys, ], f)
