import torch
from toma import toma, explicit, batchsize_cache as tbc


def raise_fake_oom():
    raise RuntimeError("CUDA out of memory.")


def test_fake_batch_none():
    batchsizes = []

    @toma.batch(initial_batchsize=64, cache_type=tbc.NoBatchsizeCache)
    def f(batchsize):
        nonlocal batchsizes
        batchsizes.append(batchsize)

        if batchsize != 16:
            raise_fake_oom()

    for _ in range(2):
        f()

    assert batchsizes == [64, 32, 16, 64, 32, 16]


def test_fake_batch_global():
    batchsizes = []

    @toma.batch(initial_batchsize=64, cache_type=tbc.GlobalBatchsizeCache)
    def f(batchsize):
        nonlocal batchsizes
        batchsizes.append(batchsize)

        if batchsize != 16:
            raise_fake_oom()

    for _ in range(2):
        f()

    assert batchsizes == [64, 32, 16, 16]


def test_fake_range_none():
    batchsizes = []

    @toma.range(initial_step=64, cache_type=tbc.NoBatchsizeCache)
    def f(start, end):
        batchsize = end - start

        nonlocal batchsizes
        batchsizes.append(batchsize)

        remaining = 128 - end

        if batchsize > 16 and batchsize > remaining:
            raise_fake_oom()

    for _ in range(2):
        f(0, 128)

    assert batchsizes == [64, 64, 32, 32, 16, 16] * 2


def test_fake_range_global():
    batchsizes = []

    @toma.range(initial_step=64, cache_type=tbc.GlobalBatchsizeCache)
    def f(start, end):
        batchsize = end - start

        nonlocal batchsizes
        batchsizes.append(batchsize)

        remaining = 128 - end

        if batchsize > 16 and batchsize > remaining:
            raise_fake_oom()

    for _ in range(2):
        f(0, 128)

    assert batchsizes == [64, 64, 32, 32, 16, 16] + [16] * 8


def test_fake_batch_none_execute():
    batchsizes = []

    @toma.batch(initial_batchsize=64, cache_type=tbc.NoBatchsizeCache)
    def f(batchsize):
        nonlocal batchsizes
        batchsizes.append(batchsize)

        if batchsize != 16:
            raise_fake_oom()

    for _ in range(2):
        f()

    assert batchsizes == [64, 32, 16, 64, 32, 16]


def test_fake_batch_global_execute():
    batchsizes = []

    for _ in range(2):

        @toma.execute.batch(initial_batchsize=64, cache_type=tbc.GlobalBatchsizeCache)
        def f(batchsize):
            nonlocal batchsizes
            batchsizes.append(batchsize)

            if batchsize != 16:
                raise_fake_oom()

    assert batchsizes == [64, 32, 16, 16]


def test_fake_range_none_execute():
    batchsizes = []

    for _ in range(2):

        @toma.execute.range(0, 128, initial_step=64, cache_type=tbc.NoBatchsizeCache)
        def f(start, end):
            batchsize = end - start

            nonlocal batchsizes
            batchsizes.append(batchsize)

            remaining = 128 - end

            if batchsize > 16 and batchsize > remaining:
                raise_fake_oom()

    assert batchsizes == [64, 64, 32, 32, 16, 16] * 2


def test_fake_range_global_execute():
    batchsizes = []

    for _ in range(2):

        @toma.execute.range(0, 128, initial_step=64, cache_type=tbc.GlobalBatchsizeCache)
        def f(start, end):
            batchsize = end - start

            nonlocal batchsizes
            batchsizes.append(batchsize)

            remaining = 128 - end

            if batchsize > 16 and batchsize > remaining:
                raise_fake_oom()

    assert batchsizes == [64, 64, 32, 32, 16, 16] + [16] * 8


def test_chunked():
    @toma.chunked(initial_step=32)
    def func(tensor, start, end):
        tensor[:] = 1.0

    tensor = torch.zeros((128, 4, 4))
    func(tensor)
    assert torch.allclose(tensor, torch.tensor(1.0))
