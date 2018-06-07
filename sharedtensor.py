import numpy as np
import ctypes
import multiprocessing as mp
import torch
from torch.autograd import Variable

def shm_as_tensor(mp_array, shape = None):
    '''
    Given a multiprocessing.Array, returns an ndarray pointing to the same data.
    '''
    if mp_array._type_ == ctypes.c_float:
        result = torch.FloatTensor(np.asarray(np.frombuffer(mp_array, dtype=np.float32)))
    elif mp_array._type_ == ctypes.c_long:
        result = torch.LongTensor(np.asarray(np.frombuffer(mp_array, dtype=np.int64)))
    else:
        print('only support float32 or int64')

    
    if shape is not None:
        result = result.view(*shape)

    return result

def tensor_to_shm(array, data_type='float32', lock = False):
    '''
    Generate an 1D multiprocessing.Array containing the data from the passed ndarray.  
    The data will be *copied* into shared memory.
    '''
    array1d = array.view(array.numel())
    if data_type == 'float32':
        c_type = ctypes.c_float
    elif data_type == 'int64':
        c_type = ctypes.c_long
    result = mp.Array(c_type, array.numel(), lock = lock)
    shm_as_tensor(result)[:] = array1d
    return result

class SharedTensor(object):
    def __init__(self, shape, dtype='float32'):
        if dtype == 'float32':
            self.shm_array = tensor_to_shm(torch.zeros(*shape))
        elif dtype == 'int64':
            self.shm_array = tensor_to_shm(torch.LongTensor(*shape).zero_(), data_type='int64')
        else:
            print('only support float32 and int64')
            exit(0)


        if len(shape) > 1:
            self.shm_tensor = shm_as_tensor(self.shm_array, shape=shape)
        else:
            self.shm_tensor = shm_as_tensor(self.shm_array)

        self.flag = mp.Value('i', 0)

    def recv(self):
        while self.flag.value == 0:
            pass
        output = self.shm_tensor.clone()
        self.flag.value = 0
        return output

    def send(self, tensor):
        while self.flag.value == 1:
            pass
        self.shm_tensor[:] = tensor
        self.flag.value = 1






