"""Torch Memory-Adaptive Algorithms.

Helpers to allow for OOM conditions and dynamic adaptation of "internal"
batchsizes. (Without affecting the computational ones.)
"""
import functools
import torch

import tomaa.torch_cuda_memory as tcm
import tomaa.stacktrace as tst
import tomaa.batchsize_cache as tbc


def simple(func, initial_batchsize, bind_to=None):
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


def simple_range(func, start, end, initial_step, bind_to=None):
    tcm.gc_cuda()

    stepsize = initial_step
    current = start
    while current < end:
        try:
            func(current, min(current + stepsize, end))
            current += stepsize
        except RuntimeError as exception:
            if stepsize > 1 and tcm.should_reduce_batch_size(exception):
                stepsize //= 2
                tcm.gc_cuda()
            else:
                raise


def simple_chunked(func, tensor, initial_step, dimension=0):
    def body(start, end):
        return func(
            tensor.narrow(dim=dimension, start=start, length=end - start),
            start, end)

    return simple_range(body, 0, tensor.shape[dimension], initial_step)


def toma(func=None, *, method=simple):
    if func is None:
        return functools.partial(toma, method=method)

    @functools.wraps(func)
    def wrapped(initial_batchsize, *args, **kwargs):
        return method(lambda bs: func(bs, *args, **kwargs), initial_batchsize,
                      bind_to=func)

    # TODO: update __doc__ string

    return wrapped


def toma_range(func=None, *, method=simple_range):
    if func is None:
        return functools.partial(toma_range, method=method)

    @functools.wraps(func)
    def wrapped(start, end, *args, maa_initial_step, **kwargs):
        return method(lambda start, end: func(start, end, *args, **kwargs),
                      start, end, maa_initial_step, bind_to=func)

    wrapped.__doc__ = f"""
Wrapped in maa_range: 

Expects start, end as first arguments.

Additional keyargs:
    maa_initial_step: initial step size to use.

{wrapped.__doc__}
"""

    return wrapped


def toma_chunked(func=None, *, method=simple_range):
    if func is None:
        return functools.partial(toma_chunked, method=method)

    @functools.wraps(func)
    def wrapped(tensor: torch.Tensor, *args, maa_initial_step, maa_dimension=0,
                **kwargs):
        def body(start, end):
            return func(tensor.narrow(dim=maa_dimension, start=start,
                                      length=end - start), start, end, *args,
                        **kwargs)

        return method(body, 0, tensor.shape[maa_dimension], maa_initial_step)

    wrapped.__doc__ = f"""
Wrapped in maa_chunked: 

Expects torch.Tensor as first argument.

Additional keyargs:
    maa_initial_step: initial step size to use
    maa_dimension: dimension of the tensor to chunk along

{wrapped.__doc__}
"""

    return wrapped


def explicit_toma(func, initial_batchsize, bind_to=None,
                  batchsize_cache_type=tbc.GlobalBatchsizeCache):
    bind_to = bind_to or func
    attr_name = f"maa_cache_{batchsize_cache_type.get_attr_suffix()}"
    if not hasattr(bind_to, attr_name):
        cache = batchsize_cache_type()
        setattr(bind_to, attr_name, cache)
    else:
        cache = getattr(bind_to, attr_name)

    cache.set_initial_batchsize(initial_batchsize)
    batchsize = cache.get_batchsize()

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


def explicit_toma_range(func, start, end, initial_step, bind_to=None,
                        batchsize_cache_type=tbc.GlobalBatchsizeCache):
    bind_to = bind_to or func
    attr_name = f"maa_cache_{batchsize_cache_type.get_attr_suffix()}"
    if not hasattr(bind_to, attr_name):
        cache = batchsize_cache_type()
        setattr(bind_to, attr_name, cache)
    else:
        cache = getattr(bind_to, attr_name)

    cache.set_initial_batchsize(initial_step)
    batchsize = cache.get_batchsize()

    tcm.gc_cuda()
    current = start
    while current < end:
        try:
            func(current, min(current + batchsize.get_batchsize(), end))
            current += batchsize.get_batchsize()
        except RuntimeError as exception:
            if batchsize.get_batchsize() > 1 and tcm.should_reduce_batch_size(
                    exception):
                batchsize.decrease_batchsize()
                tcm.gc_cuda()
            else:
                raise
