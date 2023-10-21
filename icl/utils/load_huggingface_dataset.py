import os.path

from datasets import load_dataset, load_from_disk

ROOT_FOLEDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_from_local(task_name, splits):
    dataset_path = os.path.join(ROOT_FOLEDER, 'datasets', task_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset_path: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    dataset = [dataset[split] for split in splits]
    return dataset


def load_huggingface_dataset_train_and_test(task_name):
    dataset = None
    if task_name == 'sst2':
        try:
            dataset = load_from_local(task_name, ['train', 'validation'])
        except FileNotFoundError:
            dataset = load_dataset('glue', 'sst2', split=['train', 'validation'])
        for i, _ in enumerate(dataset):
            dataset[i] = dataset[i].rename_column('sentence', 'text')
        # rename validation to test
    elif task_name == 'agnews':
        try:
            dataset = load_from_local(task_name, ['train', 'test'])
        except FileNotFoundError:
            dataset = load_dataset('ag_news', split=['train', 'test'])
    elif task_name == 'trec':
        try:
            dataset = load_from_local(task_name, ['train', 'test'])
        except FileNotFoundError:
            dataset = load_dataset('trec', split=['train', 'test'])
        coarse_label_name = 'coarse_label' if 'coarse_label' in dataset[
            0].column_names else 'label-coarse'
        for i, _ in enumerate(dataset):
            dataset[i] = dataset[i].rename_column(coarse_label_name, 'label')
    elif task_name == 'emo':
        try:
            dataset = load_from_local(task_name, ['train', 'test'])
        except FileNotFoundError:
            dataset = load_dataset('emo', split=['train', 'test'])
    if dataset is None:
        raise NotImplementedError(f"task_name: {task_name}")
    dataset = {'train': dataset[0], 'test': dataset[1]}
    return dataset
