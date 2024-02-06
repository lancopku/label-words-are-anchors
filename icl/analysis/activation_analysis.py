import os
import shutil
import torch
from .numpy_writer import CPUTensorBufferDict, NumpyWriter

_save_dict = None
_save_activation = False
_save_activation_grad = False
_save_debug = False
_writer_np = None

def set_save_activation(value):
    global _save_activation
    _save_activation = value

def get_save_activation():
    return _save_activation

def set_save_activation_grad(value):
    global _save_activation_grad
    _save_activation_grad = value

def get_save_activation_grad():
    return _save_activation_grad

def set_debug(value):
    global _save_debug
    _save_debug = value

def get_debug():
    return _save_debug
    
def clear_save_dict():
    _save_dict.clear()

def set_save_dict(writer_np):
    global _save_dict
    if _save_dict is not None and writer_np is not None:
        raise RuntimeError("save_dict已经存在")
    if writer_np is None:
        _save_dict = None
    else:
        _save_dict = CPUTensorBufferDict(writer_np=writer_np)

def set_writer_np(writer_np):
    global _writer_np
    if _writer_np is not None and writer_np is not None:
        raise RuntimeError("writer_np已经存在")
    _writer_np = writer_np
    set_save_dict(writer_np)

def get_writer_np():
    return _writer_np

def debug_fn(func):
    def wrapper(*args, **kwargs):
        if get_debug():
            return func(*args, **kwargs)
        else:
            return None
    return wrapper

debug_print = debug_fn(print)

def _add_tensor(name, value, save_type = 'activation', mode = None, log_interval = None):
    global _save_dict
    save_dict = _save_dict
    # 也许需要更多处理，比如处理bf16
    debug_print(f"add_{save_type} {name} {value.shape} {value.dtype}", flush=True)
    if isinstance(value, torch.Tensor):
        value = value.detach().clone()

        if mode == 'ijk->(i*j)k' or 'ij->ij': # 这里可以在把几个batch的数据拼接起来，变成（num_token, dim）（对于attention不适用，用在hidden上）
            value = value.reshape(-1, value.shape[-1])
        else:
            raise RuntimeError(f"不支持的 mode {mode}")
        
        if log_interval is not None: # 可以跳着存储，每几个token存 （对于attention不适用，用在hidden上）
            assert mode is not None, "log_interval需要mode"
            value = value[::log_interval,...]

        if value.dtype == torch.bfloat16 or value.dtype == torch.float16:
            value = value.float()
        value = value.cpu()
        value = value.numpy()
        save_dict[name].append(value)
    else:
        raise RuntimeError(f"不支持的类型 {type(value)}")

def add_activation(input, name, mode = None, log_interval = None):
    if not get_save_activation():
        return
    _add_tensor(name, input, 'activation', mode, log_interval)
    return input

def _add_activation_grad(input, name, mode = None, log_interval = None):
    if not get_save_activation_grad():
        return
    _add_tensor(name, input, 'activation_grad', mode, log_interval)

class IdentityToCatchGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, name, mode, log_interval):
        if not get_save_activation_grad():
            return input
        ctx.name = name
        ctx.mode = mode
        ctx.log_interval = log_interval
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if not get_save_activation_grad():
            return grad_output
        _add_activation_grad(grad_output, ctx.name, ctx.mode, ctx.log_interval)
        return grad_output, None, None, None
    
def add_activation_grad(input, name, mode = None, log_interval = None):
    if not name.endswith('_grad'):
        name = name + '_grad'
    if not get_save_activation_grad():
        return input
    input.requires_grad_(True)
    input = IdentityToCatchGrad.apply(input, name, mode, log_interval) # can not set requires_grad=True in this
    return input

def force_save_write_clear():
    debug_print("force_activation_write_clear")
    debug_print("save_dict", _save_dict.keys())
    for name, buffer in _save_dict.items():
        buffer._write()
    clear_save_dict()

def start_save(log_dir, save_activation = False, save_activation_grad = False, debug = False, continue_run = False, cover = False):
    assert save_activation or save_activation_grad, "没有需要保存的东西" 
    mode = 'a' if continue_run else 'w'
    writer_np = NumpyWriter(log_dir=log_dir, mode = mode, cover = cover)
    set_writer_np(writer_np)
    set_save_activation(save_activation)
    set_save_activation_grad(save_activation_grad)
    set_debug(debug)

def end_save():
    set_save_activation(False)
    set_save_activation_grad(False)
    force_save_write_clear()
    set_writer_np(None)

def get_result(log_dir, name, idxs=None, condition=None):
    writer_np = NumpyWriter(log_dir=log_dir, mode = 'r')
    return writer_np.read(name, idxs, condition)


    


    
