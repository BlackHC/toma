from toma import explicit
from toma.batchsize_cache import NoBatchsizeCache, GlobalBatchsizeCache, StacktraceMemoryBatchsizeCache


def raise_fake_oom():
    raise RuntimeError("CUDA out of memory.")


def test_fake_explicit_toma_none():
    batchsizes = []

    def f(batchsize):
        nonlocal batchsizes
        batchsizes.append(batchsize)

        if batchsize != 16:
            raise_fake_oom()

    for _i in range(2):
        explicit.batch(f, 64, toma_cache_type=NoBatchsizeCache)

    assert batchsizes == [64, 32, 16, 64, 32, 16]


def test_fake_explicit_toma_global():
    batchsizes = []

    def f(batchsize):
        nonlocal batchsizes
        batchsizes.append(batchsize)

        if batchsize != 16:
            raise_fake_oom()

    for _i in range(3):
        explicit.batch(f, 64, toma_cache_type=GlobalBatchsizeCache)

    explicit.batch(f, 64, toma_cache_type=GlobalBatchsizeCache)

    assert batchsizes == [64, 32, 16, 16, 16, 16]


def test_fake_explicit_toma_sm():
    batchsizes = []

    def f(batchsize):
        nonlocal batchsizes
        batchsizes.append(batchsize)

        if batchsize != 16:
            raise_fake_oom()

    for _i in range(3):
        explicit.batch(f, 64, toma_cache_type=StacktraceMemoryBatchsizeCache)

    explicit.batch(f, 64, toma_cache_type=StacktraceMemoryBatchsizeCache)

    assert batchsizes == [64, 32, 16, 16, 16, 64, 32, 16]


def test_fake_explicit_toma_mix():
    batchsizes = []

    def f(batchsize):
        nonlocal batchsizes
        batchsizes.append(batchsize)

        if batchsize != 16:
            raise_fake_oom()

    for _ in range(3):
        explicit.batch(f, 64, toma_cache_type=GlobalBatchsizeCache)

    explicit.batch(f, 64, toma_cache_type=GlobalBatchsizeCache)

    for _ in range(2):
        explicit.batch(f, 64, toma_cache_type=StacktraceMemoryBatchsizeCache)

    assert batchsizes == [64, 32, 16, 16, 16, 16, 64, 32, 16, 16]


def test_fake_explicit_toma_range_none():
    batchsizes = []

    def f(start, end):
        batchsize = end - start

        nonlocal batchsizes
        batchsizes.append(batchsize)

        remaining = 128 - end

        if batchsize > 16 and batchsize > remaining:
            raise_fake_oom()

    for _ in range(2):
        explicit.range(f, 0, 128, 64, toma_cache_type=NoBatchsizeCache)

    assert batchsizes == [64, 64, 32, 32, 16, 16] * 2


def test_fake_explicit_toma_range_global():
    batchsizes = []

    def f(start, end):
        batchsize = end - start

        nonlocal batchsizes
        batchsizes.append(batchsize)

        remaining = 128 - end

        if batchsize > 16 and batchsize > remaining:
            raise_fake_oom()

    for _ in range(2):
        explicit.range(f, 0, 128, 64, toma_cache_type=GlobalBatchsizeCache)

    explicit.range(f, 0, 128, 64, toma_cache_type=GlobalBatchsizeCache)

    assert batchsizes == [64, 64, 32, 32, 16, 16] + [16] * 8 * 2


def test_fake_explicit_toma_range_sm():
    batchsizes = []

    def f(start, end):
        batchsize = end - start

        nonlocal batchsizes
        batchsizes.append(batchsize)

        remaining = 128 - end

        if batchsize > 16 and batchsize > remaining:
            raise_fake_oom()

    for _ in range(2):
        explicit.range(f, 0, 128, 64, toma_cache_type=StacktraceMemoryBatchsizeCache)

    explicit.range(f, 0, 128, 64, toma_cache_type=StacktraceMemoryBatchsizeCache)

    assert batchsizes == [64, 64, 32, 32, 16, 16] + [16] * 8 + [64, 64, 32, 32, 16, 16]


def test_fake_explicit_toma_range_sm():
    batchsizes = []

    def f(start, end):
        batchsize = end - start

        nonlocal batchsizes
        batchsizes.append(batchsize)

        remaining = 128 - end

        if batchsize > 16 and batchsize > remaining:
            raise_fake_oom()

    for _ in range(2):
        explicit.range(f, 0, 128, 64, toma_cache_type=GlobalBatchsizeCache)

    explicit.range(f, 0, 128, 64, toma_cache_type=GlobalBatchsizeCache)

    for _ in range(2):
        explicit.range(f, 0, 128, 64, toma_cache_type=StacktraceMemoryBatchsizeCache)

    assert batchsizes == ([64, 64, 32, 32, 16, 16] + [16] * 8 * 2 + [64, 64, 32, 32, 16, 16] + [16] * 8)
