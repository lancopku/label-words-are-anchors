import os
import pickle
import warnings
from dataclasses import field, dataclass
from typing import List, Optional
from ..project_constants import FOLDER_ROOT


def set_default_to_empty_string(v, default_v, activate_flag):
    if ((default_v is not None and v == default_v) or (
            default_v is None and v is None)) and (activate_flag):
        return ""
    else:
        return f'_{v}'


@dataclass
class DeepArgs:
    task_name: str = "sst2"
    model_name: str = "gpt2-xl"
    seeds: List[int] = field(default_factory=lambda: [42])
    sample_size: int = 1000
    demonstration_shot: int = 1
    demonstration_from: str = 'train'
    demonstration_total_shot: int = None
    sample_from: str = 'test'
    device: str = 'cuda:0'
    batch_size: int = 1
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'deep')
    using_old: bool = False

    @property
    def save_file_name(self):
        file_name = (
            f"{self.task_name}_{self.model_name}_{self.demonstration_shot}_{self.demonstration_from}"
            f"_{self.sample_from}_{self.sample_size}_{'_'.join([str(seed) for seed in self.seeds])}")
        file_name += set_default_to_empty_string(self.demonstration_total_shot, None,
                                                 self.using_old)
        file_name = os.path.join(self.save_folder, file_name)
        return file_name

    def __post_init__(self):
        assert self.demonstration_from in ['train']
        assert self.sample_from in ['test']
        assert self.task_name in ['sst2', 'agnews', 'trec', 'emo']
        assert self.model_name in ['gpt2-xl', 'gpt-j-6b']
        assert 'cuda:' in self.device
        self.gpu = int(self.device.split(':')[-1])
        self.actual_sample_size = self.sample_size

        if self.task_name == 'sst2':
            label_dict = {0: ' Negative', 1: ' Positive'}
        elif self.task_name == 'agnews':
            label_dict = {0: ' World', 1: ' Sports', 2: ' Business', 3: ' Technology'}
        elif self.task_name == 'trec':
            label_dict = {0: ' Abbreviation', 1: ' Entity', 2: ' Description', 3: ' Person',
                          4: ' Location',
                          5: ' Number'}
        elif self.task_name == 'emo':
            label_dict = {0: ' Others', 1: ' Happy', 2: ' Sad', 3: ' Angry'}
        else:
            raise NotImplementedError(f"task_name: {self.task_name}")
        self.label_dict = label_dict

    def load_result(self):
        with open(self.save_file_name, 'rb') as f:
            return pickle.load(f)


@dataclass
class ReweightingArgs(DeepArgs):
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'reweighting')
    lr: float = 0.1
    train_num_per_class: int = 4
    epoch_num: int = 10
    n_head: int = 25  # 25

    def __post_init__(self):
        super(ReweightingArgs, self).__post_init__()
        save_folder = os.path.join(self.save_folder,
                                   f"lr_{self.lr}_train_num_{self.train_num_per_class}_epoch_{self.epoch_num}"
                                   f"_nhead_{self.n_head}")
        self.save_folder = save_folder


@dataclass
class CompressArgs(DeepArgs):
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'compress')

@dataclass
class CompressTopArgs(DeepArgs):
    ks_num: int = 20
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'compress_top')


@dataclass
class CompressTimeArgs(DeepArgs):
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'compress_time')


@dataclass
class AttrArgs(DeepArgs):
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'attr')


@dataclass
class ShallowArgs(DeepArgs):
    mask_layer_num: int = 5
    mask_layer_pos: str = 'first'  # first, last
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'shallow')

    @property
    def save_file_name(self):
        file_name = (
            f"{self.task_name}_{self.model_name}_{self.demonstration_shot}_{self.demonstration_from}"
            f"_{self.sample_from}_{self.sample_size}_{'_'.join([str(seed) for seed in self.seeds])}"
            f'_{self.mask_layer_num}_{self.mask_layer_pos}')
        file_name += set_default_to_empty_string(self.demonstration_total_shot, None,
                                                 self.using_old)

        file_name = os.path.join(self.save_folder, file_name)
        return file_name

    def __post_init__(self):
        super().__post_init__()
        assert self.mask_layer_pos in ['first', 'last']
        if self.mask_layer_num < 0:
            warnings.warn(f"mask_layer_num: {self.mask_layer_num} < 0!")


@dataclass
class NClassificationArgs(DeepArgs):
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'nclassfication')

@dataclass
class ShallowNonLabelArgs(ShallowArgs):
    save_folder: str = os.path.join(FOLDER_ROOT, 'results', 'shallow_non_label')