import datasets

format_s_dict = {
    'sst2': 'Review: {text}\nSentiment:{label}',
    'agnews': 'Article: {text}\nAnswer:{label}',
    'trec': 'Question: {question}\nAnswer Type:{label}',
    'emo': 'Dialogue: {text}\nEmotion:{label}',
}


def sst2_wrap_data(demonstrations, input_sample, label_dict):
    format_s = format_s_dict['sst2']
    prompts = [format_s.format(text=sample['text'], label=label_dict[sample['label']]) for
               sample in demonstrations]
    inputs = format_s.format(text=input_sample['text'], label="")
    if len(prompts) > 0:
        inputs = "\n".join(prompts + [inputs])
    return inputs


def trec_wrap_data(demonstrations, input_sample, label_dict):
    format_s = format_s_dict['trec']
    prompts = [format_s.format(question=sample['text'], label=label_dict[sample['label']]) for
               sample in demonstrations]
    inputs = format_s.format(question=input_sample['text'], label="")
    if len(prompts) > 0:
        inputs = "\n".join(prompts + [inputs])
    return inputs


def emo_wrap_data(demonstrations, input_sample, label_dict):
    format_s = format_s_dict['emo']
    prompts = [format_s.format(text=sample['text'], label=label_dict[sample['label']]) for
               sample in demonstrations]
    inputs = format_s.format(text=input_sample['text'], label="")
    if len(prompts) > 0:
        inputs = "\n".join(prompts + [inputs])
    return inputs


def agnews_wrap_data(demonstrations, input_sample, label_dict):
    format_s = format_s_dict['agnews']
    prompts = [format_s.format(text=sample['text'],
                               label=label_dict[sample['label']]) for
               sample in demonstrations]
    inputs = format_s.format(text=input_sample['text'],
                             label="")
    if len(prompts) > 0:
        inputs = "\n".join(prompts + [inputs])
    return inputs


def wrap_data(demonstrations, input_sample, label_dict, task_name):
    if task_name == 'sst2':
        return sst2_wrap_data(demonstrations, input_sample, label_dict)
    elif task_name == 'agnews':
        return agnews_wrap_data(demonstrations, input_sample, label_dict)
    elif task_name == 'trec':
        return trec_wrap_data(demonstrations, input_sample, label_dict)
    elif task_name == 'emo':
        return emo_wrap_data(demonstrations, input_sample, label_dict)
    else:
        raise NotImplementedError(f"task_name: {task_name}")


def instruct_wrapper(instruct: str, input_sample, label_dict, task_name):
    inputs = wrap_data(demonstrations=[], input_sample=input_sample, label_dict=label_dict,
                       task_name=task_name)
    format_s = '{instruct}\n{text}'
    inputs = format_s.format(text=inputs,
                             instruct=instruct)
    return inputs


def wrap_dataset(dataset: datasets.arrow_dataset.Dataset, demonstration, label_dict, task_name):
    def wrap(example):
        example['sentence'] = wrap_data(demonstrations=demonstration, input_sample=example,
                                        label_dict=label_dict, task_name=task_name)
        example['labels'] = example['label']
        return example

    dataset = dataset.map(wrap)
    return dataset


def wrap_dataset_with_instruct(dataset: datasets.arrow_dataset.Dataset, instruct, label_dict,
                               task_name):
    def wrap(example):
        example['sentence'] = instruct_wrapper(instruct=instruct, input_sample=example,
                                               label_dict=label_dict, task_name=task_name)
        example['labels'] = example['label']
        return example

    dataset = dataset.map(wrap)
    return dataset


# you may add your tokenizer's name or local path (corresponding to tokenizer.name_or_path)
# to this dict, and the corresponding model max length
default_max_length_dict = {
    'gpt2': 1024,
}


def get_max_length(tokenizer):
    if tokenizer.name_or_path in default_max_length_dict:
        return default_max_length_dict[tokenizer.name_or_path]
    max_length = tokenizer.max_len_single_sentence
    if max_length > 10000000:
        max_length = tokenizer.model_max_length
    if max_length > 10000000:
        raise ValueError(
            f"Your tokenizer has a very large `max_len_single_sentence` value: {max_length}, "
            f"you may add this to tokenizer's config, or add it to `default_max_length_dict` above")
    return max_length


def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding=True,
                         max_length=get_max_length(tokenizer),
                         truncation=True,
                         return_tensors='pt')

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets


def remove_str_columns(dataset):
    remove_keys = {k for k, v in dataset.features.items() if v.dtype == 'string'}
    dataset = dataset.remove_columns(list(remove_keys))
    return dataset
