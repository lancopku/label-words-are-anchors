import os.path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_PATH_LIST = ['Your_path'] # add your model path if you want to load local models



def convert_path_old(path: str, ROOT_PATH, load_type: str) -> str:
    assert load_type in ['tokenizer', 'model']
    return os.path.join(ROOT_PATH, load_type + 's', path)

def convert_path(path: str, ROOT_PATH, load_type: str) -> str:
    assert load_type in ['tokenizer', 'model']
    return os.path.join(ROOT_PATH, path)

def load_local_model_or_tokenizer(model_name: str, load_type: str):
    if load_type in 'tokenizer':
        LoadClass = AutoTokenizer
    elif load_type in 'model':
        LoadClass = AutoModelForCausalLM
    else:
        raise ValueError(f'load_type: {load_type} is not supported')

    model = None
    for ROOT_PATH in ROOT_PATH_LIST:
        try:
            folder_path = convert_path_old(model_name, ROOT_PATH, load_type)
            if not os.path.exists(folder_path):
                continue
            print(f'loading {model_name} {load_type} from {folder_path} ...')
            model = LoadClass.from_pretrained(folder_path)
            print('finished loading')
            break
        except:
            continue
    if model is not None:
        return model
    for ROOT_PATH in ROOT_PATH_LIST:
        try:
            folder_path = convert_path(model_name, ROOT_PATH, load_type)
            if not os.path.exists(folder_path):
                continue
            print(f'loading {model_name} {load_type} from {folder_path} ...')
            model = LoadClass.from_pretrained(folder_path)
            print('finished loading')
            break
        except:
            continue
    return model

def get_model_layer_num(model = None, model_name = None):
    num_layer = None
    if model is not None:
        if hasattr(model.config, 'num_hidden_layers'):
            num_layer = model.config.num_hidden_layers
        elif hasattr(model.config, 'n_layers'):
            num_layer = model.config.n_layers
        elif hasattr(model.config, 'n_layer'):
            num_layer = model.config.n_layer
        else:
            pass
    elif model_name is not None:
        pass
    if num_layer is None:
        raise ValueError(f'cannot get num_layer from model: {model} or model_name: {model_name}')
    return num_layer
