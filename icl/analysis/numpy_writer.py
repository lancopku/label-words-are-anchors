import os

import numpy as np

_numpy_writer_debug = False

def set_writer_np(writer_np):
    global _writer_np
    _writer_np = writer_np

def get_writer_np():
    return _writer_np

def set_numpy_writer_debug(value):
    global _numpy_writer_debug
    _numpy_writer_debug = value

def get_numpy_writer_debug():
    return _numpy_writer_debug

def debug_fn(func):
    def wrapper(*args, **kwargs):
        if get_numpy_writer_debug():
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


debug_print = debug_fn(print)

class NumpyWriter:
    def __init__(self, log_dir, mode = 'w', cover = False):
        '''
        mode:   'w' for write
                'a' for append (for case you run half of the experiment and want to continue) 
                'r' for read
        cover: if True, remove the log_dir if it exists
        '''
        assert mode in ['w', 'a', 'r'], f"mode {mode} not allowed"
        self.log_dir = log_dir
        assert not (mode == 'r' and cover), "mode == 'r' and cover == True not allowed"
        if not cover and mode == 'w':
            assert not os.path.exists(log_dir), f"log_dir {log_dir} ({os.path.abspath(log_dir)}) already exists"
        if cover:
            if os.path.exists(log_dir):
                import shutil
                shutil.rmtree(log_dir)
                print(f"remove {log_dir}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.mode = mode
        self.cover = cover

    
    def fps(self, name, mode):
        # here we only use 'a' and 'r', since activation for different samples should all be saved
        if mode == 'r' and self.mode in ['w','a']:
            raise RuntimeError(f"You are use a writer to read")
        if mode == 'a' and self.mode == 'r':
            raise RuntimeError(f"You are use a reader to write")
        if mode == 'a':
            return open(os.path.join(self.log_dir, name), "ab")
        elif mode == 'r':
            return open(os.path.join(self.log_dir, name), "rb")
        else:
            raise RuntimeError(f"mode {mode} not allowed")

    def append_tensor(self, tensor, fp, debug=False):
        tensor_numpy = tensor.numpy() if not isinstance(tensor, np.ndarray) else tensor
        np.save(fp, np.array(tensor_numpy.shape))
        # 保存 tensor 数据
        np.save(fp, tensor_numpy)

        debug_print(
            f"append_tensor {tensor.shape} {tensor.dtype} {tensor_numpy.shape} {tensor_numpy.dtype}", flush=True)

    def read_tensors(self, fp, idxs=None, condition=None):
        assert not (
            idxs is not None and condition is not None), "idxs 和 condition 不能同时设置"
        if idxs is not None:
            def condition(idx): return idx in idxs
        elif condition is None:
            def condition(idx): return True
        tensors = []
        idx = 0
        while True:
            try:
                # 读取形状信息
                shape = np.load(fp, allow_pickle=True)
                # 读取 tensor 数据
                tensor = np.load(fp, allow_pickle=True).reshape(shape)
                if condition(idx):
                    tensors.append(tensor)
                idx += 1
            except Exception as e:
                # Check if the cause of the UnpicklingError is an EOFError
                if isinstance(e.__cause__, EOFError):
                    break
                else:
                    raise
        return tensors

    def write(self, data, name, global_step=None, multiple_tensors=True, debug=False):
        with self.fps(name,mode='a') as fp:
            if multiple_tensors:
                for tensor in data:
                    self.append_tensor(tensor, fp, debug=debug)
            else:
                self.append_tensor(data, fp, debug=debug)
        debug_print(f"write {name}", flush=True)

    def read(self, name, idxs=None, condition=None):
        with self.fps(name,mode='r') as fp:
            return self.read_tensors(fp, idxs=idxs, condition=condition)

class CPUTensorBuffer(list):
    def __init__(self, tensors, cpu_buffer=100, allow_auto_write=True, writer_np=None, name=None):
        super().__init__(tensors)
        self.cpu_buffer_nbytes = cpu_buffer * 1024 * 1024
        self.mem = 0
        for tensor in self:
            self.mem += tensor.nbytes
        self.allow_auto_write = allow_auto_write
        self.writer_np = writer_np
        self.name = name
        if self.allow_auto_write:
            assert self.writer_np is not None, "writer_np must be set when allow_auto_write is True"
            assert self.name is not None, "name must be set when allow_auto_write is True"

    def clear(self) -> None:
        self.mem = 0
        # 没想好要不要手动删tensor（已经放到cpu上了）
        return super().clear()

    def _write(self):
        self.writer_np.write(self, self.name, multiple_tensors=True)
        self.clear()

    def auto_check_write(func):
        def wrapper(self, *args, **kwargs):
            if self.allow_auto_write:
                if self.mem >= self.cpu_buffer_nbytes:
                    self._write()
            return func(self, *args, **kwargs)
        return wrapper

    @auto_check_write
    def append(self, __object) -> None:
        self.mem += __object.nbytes
        return super().append(__object)

    @auto_check_write
    def extend(self, __iterable) -> None:
        for obj in __iterable:
            self.mem += obj.nbytes
        return super().extend(__iterable)


class CPUTensorBufferDict(dict):
    def __init__(self, writer_np):
        super().__init__()
        assert isinstance(writer_np, NumpyWriter)
        self.writer_np = writer_np

    # 暂时应该保证所有的CPUTensorBuffer.writer_np都和__init__里的writer_np一样
    def __missing__(self, key):
        default_value = CPUTensorBuffer(
            [], allow_auto_write=True, cpu_buffer=100,  writer_np=self.writer_np, name=key)
        self[key] = default_value
        return default_value
