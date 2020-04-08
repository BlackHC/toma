import torch
import pytest_benchmark

# Preload this import
import resource

from toma import simple, toma, explicit, NoBatchsizeCache


def test_toma(benchmark):
    benchmark.extra_info["Debug Mode"] = __debug__

    @toma.chunked(initial_step=32)
    def func(tensor, start, end):
        tensor[:] = 1.0

    tensor = torch.zeros((128, 256, 256))
    benchmark(func, tensor)


def test_toma_no_cache(benchmark):
    benchmark.extra_info["Debug Mode"] = __debug__

    @toma.chunked(initial_step=32, cache_type=NoBatchsizeCache)
    def func(tensor, start, end):
        tensor[:] = 1.0

    tensor = torch.zeros((128, 256, 256))
    benchmark(func, tensor)


def test_explicit(benchmark):
    benchmark.extra_info["Debug Mode"] = __debug__

    def func(tensor, start, end):
        tensor[:] = 1.0

    tensor = torch.zeros((128, 256, 256))
    benchmark(explicit.chunked, func, tensor, 32)


def test_simple(benchmark):
    benchmark.extra_info["Debug Mode"] = __debug__

    def func(tensor, start, end):
        tensor[:] = 1.0

    tensor = torch.zeros((128, 256, 256))
    benchmark(simple.chunked, func, tensor, 32)


def test_native(benchmark):
    benchmark.extra_info["Debug Mode"] = __debug__

    def func(tensor, start, end):
        tensor[:] = 1.0

    def native(func, tensor, batch):
        end = tensor.shape[0]
        current = 0
        while current < end:
            current_end = min(current + batch, end)
            func(tensor.narrow(0, current, current_end - current), current, current_end)
            current = current_end

    tensor = torch.zeros((128, 256, 256))
    benchmark(native, func, tensor, 32)
