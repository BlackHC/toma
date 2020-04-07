"""Torch Memory-Adaptive Algorithms.

Helpers to allow for OOM conditions and dynamic adaptation of "internal"
batchsizes. (Without affecting the computational ones.)
"""
import functools
import torch

import tomma.torch_cuda_memory as tcm
import tomma.stacktrace as tst
import tomma.batchsize_cache as tbc


def simple_mma(func, initial_batchsize):
    tcm.gc_cuda()

    batchsize = initial_batchsize
    while True:
        try:
            return func(batchsize)
        except RuntimeError as exception:
            if batchsize > 1 and tcm.should_reduce_batch_size(exception):
                batchsize //= 2
                tcm.gc_cuda()
            else:
                raise


def simple_mma_range(body, start, end, initial_step):
    tcm.gc_cuda()

    stepsize = initial_step
    current = start
    while current < end:
        try:
            body(current, min(current + stepsize, end))
            current += stepsize
        except RuntimeError as exception:
            if stepsize > 1 and tcm.should_reduce_batch_size(exception):
                stepsize //= 2
                tcm.gc_cuda()
            else:
                raise


def mma(func=None, *, method=simple_mma):
    if func is None:
        return functools.partial(mma, method=method)

    @functools.wraps(func)
    def wrapped(initial_batchsize, *args, **kwargs):
        return method(lambda bs: func(bs, *args, **kwargs), initial_batchsize)

    # TODO: update __doc__ string

    return wrapped


def mma_range(func=None, *, method=simple_mma_range):
    if func is None:
        return functools.partial(mma_range, method=method)

    @functools.wraps(func)
    def wrapped(start, end, *args, mma_initial_step, **kwargs):
        return method(func, start, end, mma_initial_step)

    wrapped.__doc__ = f"""
Wrapped in mma_range: 

Expects start, end as first arguments.

Additional keyargs:
    mma_initial_step: initial step size to use.

{wrapped.__doc__}
"""

    return wrapped


def mma_chunked(func=None, *, method=simple_mma_range):
    if func is None:
        return functools.partial(mma_chunked, method=method)

    @functools.wraps(func)
    def wrapped(tensor: torch.Tensor, *args, mma_initial_step, mma_dimension=0, **kwargs):
        def body(current, current_end):
            return func(tensor.narrow(dim=mma_dimension, start=current, length=current_end - current), *args, **kwargs)

        return method(body, 0, tensor.shape[mma_dimension], mma_initial_step)

    wrapped.__doc__ = f"""
Wrapped in mma_chunked: 

Expects torch.Tensor as first argument.

Additional keyargs:
    mma_initial_step: initial step size to use
    mma_dimension: dimension of the tensor to chunk along

{wrapped.__doc__}
"""

    return wrapped


def explicit_mma(func, initial_batchsize, batchsize_cache=tbc.GlobalBatchsizeCache):
    if not hasattr(func, "mma_cache"):
        func.mma_cache = batchsize_cache()

    func.mma_cache.set_initial_batchsize(initial_batchsize)
    batchsize = func.mma_cache.get_batchsize()

    while True:
        try:
            value = batchsize.get_batchsize()
            return func(value)
        except RuntimeError as exception:
            if value > 1 and tcm.should_reduce_batch_size(exception):
                batchsize.decrease_batchsize()
                tcm.gc_cuda()
            else:
                raise


def explicit_mma_range(func, start, end, initial_step, batchsize_cache=tbc.GlobalBatchsizeCache):
    if not hasattr(func, "mma_cache"):
        func.mma_cache = batchsize_cache()

    func.mma_cache.set_initial_batchsize(initial_step)
    batchsize = func.mma_cache.get_batchsize()

    tcm.gc_cuda()
    current = start
    while current < end:
        try:
            func(current, min(current + batchsize.get_batchsize(), end))
            current += batchsize.get_batchsize()
        except RuntimeError as exception:
            if batchsize.get_batchsize() > 1 and tcm.should_reduce_batch_size(exception):
                batchsize.decrease_batchsize()
                tcm.gc_cuda()
            else:
                raise
