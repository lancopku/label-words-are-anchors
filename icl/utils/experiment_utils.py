import os
import pickle
import warnings
from time import sleep
from typing import *
import numpy as np
from copy import deepcopy, copy
import random


def get_gpu_ids(gpu_num=100, gpu_ids=None, min_memory: float = 5000):
    if gpu_ids is None:
        gpu_ids = []
    os.makedirs("./buffer", exist_ok=True)
    auto_num = gpu_num - (0 if gpu_ids is None else len(gpu_ids))
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./buffer/tmp')
    memory_gpu = [int(x.split()[2]) for x in open('./buffer/tmp', 'r').readlines()]
    enough_memory_gpu_ids = [(idx, _) for idx, _ in enumerate(memory_gpu) if _ > min_memory]
    enough_memory_gpu_ids = sorted(enough_memory_gpu_ids, key=lambda x: x[1], reverse=True)
    auto_gpu_ids = [idx for idx, _ in
                    enough_memory_gpu_ids[0:max(auto_num, len(enough_memory_gpu_ids))]]
    gpu_ids.extend(auto_gpu_ids)
    return gpu_ids


def set_gpu(gpus: Union[List[int], int]):
    gpus = [str(gpus)] if isinstance(gpus, int) else [str(_) for _ in gpus]
    gpus = ",".join(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


class Combination_iterater:
    def __init__(self, hyperparameter_lists: Dict[str, List[Any]]):
        self.hyperparameter_lists: Dict[str, List[Any]] = hyperparameter_lists
        self.hyperparameter_list_lengths: List[int] = [len(_) for _ in
                                                       hyperparameter_lists.values()]
        self.length = int(np.prod(self.hyperparameter_list_lengths))
        self.step: int = 0

    def get_combination_decode_int2indexs(self, i: int) -> List[int]:
        assert i < self.length
        indexs = []
        for _ in self.hyperparameter_list_lengths:
            indexs.append(i % _)
            i //= _
        return indexs

    def get_combination(self, i: int) -> Dict[str, Any]:
        indexs = self.get_combination_decode_int2indexs(i)
        hyperparameters = {k: v[index] for (k, v), index in
                           zip(self.hyperparameter_lists.items(), indexs)}
        return hyperparameters

    def __next__(self) -> Dict[str, Any]:
        if self.step >= self.length:
            raise StopIteration
        hyperparameters = self.get_combination(self.step)
        self.step += 1
        return hyperparameters

    def __iter__(self):
        return self

    def __len__(self):
        return self.length - self.step


class Gpu_assinger:
    def __init__(self, gpu_list: List[int], min_memory: float = 0):
        if gpu_list is None:
            gpu_list = get_gpu_ids(min_memory=min_memory)
        self.gpu_list = gpu_list
        self.free_gpu_list = copy(gpu_list)
        warnings.warn('free gpu list is deprecated, use gpu_list instead')
        print(f"free gpu list: {self.free_gpu_list}")

    def get(self) -> int:
        if len(self.free_gpu_list) > 0:
            free_gpu = self.free_gpu_list.pop(0)
            return free_gpu
        else:
            return -1

    def add(self, free_gpu: int) -> None:
        self.free_gpu_list.append(free_gpu)


class Sh_run:
    def __init__(self, hyperparameter_lists: List[Dict[str, List[Any]]],
                 gpu_list: List[int] = None, default_sh_file="./gpu_sh.sh") -> None:
        self.experiment_strs = [_['expr'] for _ in hyperparameter_lists]
        for _ in hyperparameter_lists:
            _.pop('expr')
        self.hyper_combinator_iterators = [
            Combination_iterater(hyperparameter_lists=hyperparameter_list) for hyperparameter_list
            in hyperparameter_lists]
        self.gpu_assigner = Gpu_assinger(gpu_list=gpu_list)
        self.gpu_list = self.gpu_assigner.free_gpu_list
        self.gpu_str_dict: Dict[int, str] = {gpu: "" for gpu in self.gpu_list}
        self.default_sh_file = default_sh_file
        self.explicit_set_gpu = False

    def generate_specific_sh_file_name(self, gpu: int) -> str:
        return self.default_sh_file.replace(".sh", "%d.sh" % gpu)

    def get_args_str(self, arg_dict: Dict[str, Any]) -> str:
        s = []
        for k, v in arg_dict.items():
            if isinstance(v, bool) and v == True:
                s.append("--%s" % k)
            else:
                s.append("--%s %s" % (k, str(v)))
        s = " ".join(s)
        return s

    def run(self):
        _ = 0
        for expr, hyper_combinator_iterator in zip(self.experiment_strs,
                                                   self.hyper_combinator_iterators):
            for i, hyper_arg in enumerate(hyper_combinator_iterator):
                gpu = self.gpu_assigner.get()
                if not self.explicit_set_gpu:
                    hyper_arg["device"] = f'cuda:{gpu}'
                    prefix_str = "nohup"
                else:
                    prefix_str = f"CUDA_VISIBLE_DEVICES={gpu}"
                s = " ".join([prefix_str, expr, self.get_args_str(hyper_arg)])
                self.gpu_str_dict[gpu] += s + "\n"
                self.gpu_assigner.add(gpu)
        for k, v in self.gpu_str_dict.items():
            with open(self.generate_specific_sh_file_name(k), "w") as f:
                f.write(v)
        with open(self.default_sh_file, "w") as f:
            s = ""
            for k in self.gpu_str_dict.keys():
                s += " ".join(["nohup sh", self.generate_specific_sh_file_name(k), "&"]) + "\n"
            f.write(s)

def get_perf_from_perf_dict(perf_dict: Dict[str, Any], perf_name: Optional[str] = None):
    if perf_name is None:
        if 'acc' in perf_dict:
            perf_name = 'acc'
        elif 'f1' in perf_dict:
            perf_name = 'f1'
        elif 'f' in perf_dict:
            perf_name = 'f'
        else:
            raise ValueError("perf_name is None, and can't find proper erf_name in perf_dict")
    perf = perf_dict[perf_name]
    if isinstance(perf, list):
        perf = np.mean(perf)
    return perf
