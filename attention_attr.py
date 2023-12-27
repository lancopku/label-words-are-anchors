import pickle
import warnings
from dataclasses import dataclass, field
from typing import List
import os
import numpy as np
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
import torch
import torch.nn.functional as F

from icl.lm_apis.lm_api_base import LMForwardAPI
from icl.utils.data_wrapper import wrap_dataset, tokenize_dataset
from icl.utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from icl.utils.prepare_model_and_tokenizer import load_model_and_tokenizer, \
    get_label_id_dict_for_args
from icl.utils.random_utils import set_seed
from icl.utils.other import load_args, set_gpu, sample_two_set_with_shot_per_class
from transformers import Trainer, TrainingArguments, PreTrainedModel, AutoModelForCausalLM, \
    AutoTokenizer
from icl.utils.load_local import convert_path_old, load_local_model_or_tokenizer, \
    get_model_layer_num
from icl.util_classes.arg_classes import AttrArgs
from icl.util_classes.predictor_classes import Predictor
from transformers import HfArgumentParser
from datasets import concatenate_datasets
from datasets.utils.logging import disable_progress_bar
import icl.analysis.attentioner_for_attribution
from icl.analysis.attentioner_for_attribution import AttentionAdapter, \
    GPT2AttentionerManager
from icl.utils.other import dict_to

hf_parser = HfArgumentParser((AttrArgs,))
args: AttrArgs = hf_parser.parse_args_into_dataclasses()[0]

set_gpu(args.gpu)
if args.sample_from == 'test':
    dataset = load_huggingface_dataset_train_and_test(args.task_name)
else:
    raise NotImplementedError(f"sample_from: {args.sample_from}")

model, tokenizer = load_model_and_tokenizer(args)
args.label_id_dict = get_label_id_dict_for_args(args, tokenizer)

# model = model.half()

model = LMForwardAPI(model=model, model_name=args.model_name, tokenizer=tokenizer,
                     device='cuda:0',
                     label_dict=args.label_dict)


num_layer = get_model_layer_num(model=model.model, model_name=args.model_name)
predictor = Predictor(label_id_dict=args.label_id_dict, pad_token_id=tokenizer.pad_token_id,
                      task_name=args.task_name, tokenizer=tokenizer, layer=num_layer)


def prepare_analysis_dataset(seed):
    if args.sample_from == 'test':
        if len(dataset['test']) < args.actual_sample_size:
            args.actual_sample_size = len(dataset['test'])
            warnings.warn(
                f"sample_size: {args.sample_size} is larger than test set size: {len(dataset['test'])},"
                f"actual_sample_size is {args.actual_sample_size}")
        test_sample = dataset['test'].shuffle(seed=seed).select(range(args.actual_sample_size))
    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")
    disable_progress_bar()
    demonstration = dataset['train']
    class_num = len(set(demonstration['label']))
    np_labels = np.array(demonstration['label'])
    ids_for_demonstrations = [np.where(np_labels == class_id)[0] for class_id in range(class_num)]
    demonstrations_contexted = []
    np.random.seed(seed)
    for i in range(len(test_sample)):
        demonstration_part_ids = []
        for _ in ids_for_demonstrations:
            demonstration_part_ids.extend(np.random.choice(_, args.demonstration_shot))
        demonstration_part = demonstration.select(demonstration_part_ids)
        demonstration_part = demonstration_part.shuffle(seed=seed)
        demonstration_part = wrap_dataset(test_sample.select([i]), demonstration_part,
                                          args.label_dict,
                                          args.task_name)
        demonstrations_contexted.append(demonstration_part)
    demonstrations_contexted = concatenate_datasets(demonstrations_contexted)
    demonstrations_contexted = demonstrations_contexted.filter(
        lambda x: len(tokenizer(x["sentence"])['input_ids']) <= tokenizer.max_len_single_sentence)
    demonstrations_contexted = tokenize_dataset(demonstrations_contexted, tokenizer=tokenizer)
    return demonstrations_contexted


demonstrations_contexted = prepare_analysis_dataset(args.seeds[0])

if args.model_name in ['gpt2-xl']:
    attentionermanger = GPT2AttentionerManager(model.model)
else:
    raise NotImplementedError(f"model_name: {args.model_name}")

training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                  per_device_eval_batch_size=1,
                                  per_device_train_batch_size=1)
trainer = Trainer(model=model, args=training_args)
analysis_dataloader = trainer.get_eval_dataloader(demonstrations_contexted)


for p in model.parameters():
    p.requires_grad = False

def get_proportion(saliency, class_poss, final_poss):
    saliency = saliency.detach().clone().cpu()
    class_poss = torch.hstack(class_poss).detach().clone().cpu()
    final_poss = final_poss.detach().clone().cpu()
    assert len(saliency.shape) == 2 or (len(saliency.shape) == 3 and saliency.shape[0] == 1)
    if len(saliency.shape) == 3:
        saliency = saliency.squeeze(0)
    saliency = saliency.numpy()
    np.fill_diagonal(saliency, 0)
    proportion1 = saliency[class_poss, :].sum()
    proportion2 = saliency[final_poss, class_poss].sum()
    proportion3 = saliency.sum() - proportion1 - proportion2

    N = int(final_poss)
    sum3 = (N + 1) * N / 2 - sum(class_poss) - len(class_poss)
    proportion1 = proportion1 / sum(class_poss)
    proportion2 = proportion2 / len(class_poss)
    proportion3 = proportion3 / sum3
    proportions = np.array([proportion1, proportion2, proportion3])
    return proportions


pros_list = []

for idx, data in tqdm(enumerate(analysis_dataloader)):
    data = dict_to(data, model.device)
    print(data['input_ids'].shape)
    attentionermanger.zero_grad()
    output = model(**data)
    label = data['labels']
    loss = F.cross_entropy(output['logits'], label)
    loss.backward()
    class_poss, final_poss = predictor.get_pos({'input_ids': attentionermanger.input_ids})
    pros = []
    for i in range(len(attentionermanger.attention_adapters)):
        saliency = attentionermanger.grad(use_abs=True)[i]
        pro = get_proportion(saliency, class_poss, final_poss)
        pros.append(pro)
    pros = np.array(pros)
    pros = pros.T
    pros_list.append(pros)

pros_list = np.array(pros_list)

os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
with open(args.save_file_name, 'wb') as f:
    pickle.dump(pros_list, f)
