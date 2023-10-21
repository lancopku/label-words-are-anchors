import numpy as np
import random, os
import torch.backends.cudnn


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def np_temp_random(seed: int):
    def np_temp_random_inner(func):
        def np_temp_random_innner_inner(*args, **kwargs):
            ori_state = np.random.get_state()
            np.random.seed(seed)
            result = func(*args, **kwargs)
            np.random.set_state(ori_state)
            return result

        return np_temp_random_innner_inner

    return np_temp_random_inner
