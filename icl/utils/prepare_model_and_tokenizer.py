import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM

from .load_local import load_local_model_or_tokenizer
from ..util_classes.arg_classes import DeepArgs


def load_model_and_tokenizer(args: DeepArgs):
    if args.model_name in ['gpt2-xl', 'gpt-j-6b']:
        tokenizer = load_local_model_or_tokenizer(args.model_name, 'tokenizer')
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = load_local_model_or_tokenizer(args.model_name, 'model')
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise NotImplementedError(f"model_name: {args.model_name}")
    return model, tokenizer


def get_label_id_dict_for_args(args: DeepArgs, tokenizer):
    label_id_dict = {k: tokenizer.encode(v, add_special_tokens=False)[0] for k, v in
                          args.label_dict.items()}
    for v in args.label_dict.values():
        token_num = len(tokenizer.encode(v, add_special_tokens=False))
        if token_num != 1:
            warnings.warn(f"{v} in {args.task_name} has token_num: {token_num} which is not 1")
    return label_id_dict
