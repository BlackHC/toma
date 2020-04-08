"""Torch Memory-Adaptive Algorithms.

Helpers to allow for OOM conditions and dynamic adaptation of "internal"
batchsizes. (Without affecting the computational ones.)
"""
import functools
from typing import Type, Optional

import torch

import toma.stacktrace as tst
from toma.batchsize_cache import StacktraceMemoryBatchsizeCache, NoBatchsizeCache, GlobalBatchsizeCache
from toma.cpu_memory import is_out_of_cpu_memory
from toma.torch_cuda_memory import is_cuda_out_of_memory, is_cudnn_snafu, gc_cuda


DEFAULT_CACHE_TYPE = StacktraceMemoryBatchsizeCache


class simple:
    """
    Straight-forward wrappers (can be copy-pasted and hacked easily).
    """

    @staticmethod
    def batch(func, initial_batchsize: int, *args, **kwargs):
        gc_cuda()

        batchsize = initial_batchsize
        while True:
            try:
                return func(batchsize, *args, **kwargs)
            except RuntimeError as exception:
                if batchsize > 1 and should_reduce_batch_size(exception):
                    batchsize //= 2
                    gc_cuda()
                else:
                    raise

    @staticmethod
    def range(func, start: int, end: int, initial_step: int, *args, **kwargs):
        gc_cuda()

        stepsize = initial_step
        current = start
        while current < end:
            try:
                func(current, min(current + stepsize, end), *args, **kwargs)
                current += stepsize
            except RuntimeError as exception:
                if stepsize > 1 and should_reduce_batch_size(exception):
                    stepsize //= 2
                    gc_cuda()
                else:
                    raise

    @staticmethod
    def chunked(func, tensor, initial_step: int, dimension: int = 0):
        def body(start, end):
            return func(tensor.narrow(dim=dimension, start=start, length=end - start), start, end)

        return simple.range(body, 0, tensor.shape[dimension], initial_step)


class toma:
    """
    Decorators that make it easy to wrap functions.
    """

    @staticmethod
    def batch(func=None, *, initial_batchsize=None, cache_type=DEFAULT_CACHE_TYPE):
        if func is None:
            return functools.partial(toma.batch, initial_batchsize=initial_batchsize, cache_type=cache_type)

        @functools.wraps(func)
        def wrapped(*args, toma_initial_batchsize=None, **kwargs):
            _initial_batchsize = toma_initial_batchsize or initial_batchsize
            return explicit.batch(func, _initial_batchsize, *args, toma_cache_type=cache_type, **kwargs)

        wrapped.__doc__ = f"""
Wrapped in toma.batch: 

Additional keyargs:
    toma_initial_batchsize: initial step size to use.

{wrapped.__doc__}
"""

        return wrapped

    @staticmethod
    def range(func=None, *, initial_step: Optional[int] = None, cache_type=DEFAULT_CACHE_TYPE):
        if func is None:
            return functools.partial(toma.range, initial_step=initial_step, cache_type=cache_type)

        @functools.wraps(func)
        def wrapped(start: int, end: int, *args, toma_initial_step: Optional[int] = None, **kwargs):
            _initial_step = toma_initial_step or initial_step

            return explicit.range(
                func, start, end, _initial_step, *args, toma_context=func, toma_cache_type=cache_type, **kwargs
            )

        wrapped.__doc__ = f"""
Wrapped in toma.range: 

Additional keyargs:
    toma_initial_step: initial step size to use.

{wrapped.__doc__}
"""

        return wrapped

    @staticmethod
    def chunked(
        func=None,
        *,
        initial_step: Optional[int] = None,
        dimension: Optional[int] = None,
        cache_type: Type = DEFAULT_CACHE_TYPE,
    ):
        dimension = dimension or 0
        if func is None:
            return functools.partial(
                toma.chunked, initial_step=initial_step, dimension=dimension, cache_type=cache_type
            )

        @functools.wraps(func)
        def wrapped(
            tensor: torch.Tensor,
            *args,
            toma_initial_step: Optional[int] = None,
            toma_dimension: Optional[int] = None,
            **kwargs,
        ):
            _initial_step = toma_initial_step or initial_step
            _dimension = toma_dimension or dimension

            explicit.chunked(
                func, tensor, _initial_step, *args, toma_dimension=toma_dimension, toma_cache_type=cache_type, **kwargs
            )

        wrapped.__doc__ = f"""
Wrapped in toma.chunked: 

Additional keyargs:
    toma_initial_step: initial step size to use
    toma_dimension: dimension of the tensor to chunk along

{wrapped.__doc__}
"""

        return wrapped


CONTEXT_CACHE_SIZE = 2 ** 14


@functools.lru_cache(CONTEXT_CACHE_SIZE)
def get_cache_for_context(batchsize_cache_type, context):
    return batchsize_cache_type()


class explicit:
    """
    Explicit calls that can use different cache types to memorize settings.
    """

    @staticmethod
    def batch(
        func, initial_batchsize: int, *args, toma_context=None, toma_cache_type: Type = DEFAULT_CACHE_TYPE, **kwargs
    ):
        gc_cuda()

        cache = get_cache_for_context(toma_cache_type, toma_context or func)

        batchsize = cache.get_batchsize(initial_batchsize)

        while True:
            try:
                value = batchsize.get()
                return func(value, *args, **kwargs)
            except RuntimeError as exception:
                if value > 1 and should_reduce_batch_size(exception):
                    batchsize.decrease_batchsize()
                    gc_cuda()
                else:
                    raise

    @staticmethod
    def range(
        func,
        start: int,
        end: int,
        initial_step: int,
        *args,
        toma_context=None,
        toma_cache_type: Type = DEFAULT_CACHE_TYPE,
        **kwargs,
    ):
        gc_cuda()

        cache = get_cache_for_context(toma_cache_type, toma_context or func)

        batchsize = cache.get_batchsize(initial_step)

        gc_cuda()
        current = start
        while current < end:
            try:
                func(current, min(current + batchsize.get(), end), *args, **kwargs)
                current += batchsize.get()
            except RuntimeError as exception:
                if batchsize.get() > 1 and should_reduce_batch_size(exception):
                    batchsize.decrease_batchsize()
                    gc_cuda()
                else:
                    raise

    @staticmethod
    def chunked(
        func,
        tensor: torch.Tensor,
        initial_step: int,
        *args,
        toma_dimension: int = None,
        toma_context=None,
        toma_cache_type: Type = DEFAULT_CACHE_TYPE,
        **kwargs,
    ):
        toma_dimension = toma_dimension or 0

        def body(start: int, end: int):
            return func(tensor.narrow(dim=toma_dimension, start=start, length=end - start), start, end, *args, **kwargs)

        explicit.range(
            body,
            0,
            tensor.shape[toma_dimension],
            initial_step,
            *args,
            toma_context=toma_context or func,
            toma_cache_type=toma_cache_type,
        )


def should_reduce_batch_size(exception):
    return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception) or is_out_of_cpu_memory(exception)
