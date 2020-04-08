from tomaa import explicit_toma, explicit_toma_range
from tomaa.batchsize_cache import GlobalBatchsizeCache, \
    StacktraceMemoryBatchsizeCache


def raise_fake_oom():
    raise RuntimeError("CUDA out of memory.")


def test_fake_explicit_toma_global():
    batchsizes = []

    def f(batchsize):
        nonlocal batchsizes
        batchsizes.append(batchsize)

        if batchsize != 16:
            raise_fake_oom()

    for _i in range(3):
        explicit_toma(f, 64, batchsize_cache_type=GlobalBatchsizeCache)

    explicit_toma(f, 64, batchsize_cache_type=GlobalBatchsizeCache)

    assert batchsizes == [64, 32, 16, 16, 16, 16]


def test_fake_explicit_toma_sm():
    batchsizes = []

    def f(batchsize):
        nonlocal batchsizes
        batchsizes.append(batchsize)

        if batchsize != 16:
            raise_fake_oom()

    for _i in range(3):
        explicit_toma(f, 64, batchsize_cache_type=StacktraceMemoryBatchsizeCache)

    explicit_toma(f, 64, batchsize_cache_type=StacktraceMemoryBatchsizeCache)

    assert batchsizes == [64, 32, 16, 16, 16, 64, 32, 16]


def test_fake_explicit_toma_mix():
    batchsizes = []

    def f(batchsize):
        nonlocal batchsizes
        batchsizes.append(batchsize)

        if batchsize != 16:
            raise_fake_oom()

    for _ in range(3):
        explicit_toma(f, 64, batchsize_cache_type=GlobalBatchsizeCache)

    explicit_toma(f, 64, batchsize_cache_type=GlobalBatchsizeCache)

    for _ in range(2):
        explicit_toma(f, 64, batchsize_cache_type=StacktraceMemoryBatchsizeCache)

    assert batchsizes == [64, 32, 16, 16, 16, 16, 64, 32, 16, 16]


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
        explicit_toma_range(f, 0, 128, 64,
                            batchsize_cache_type=GlobalBatchsizeCache)

    explicit_toma_range(f, 0, 128, 64, batchsize_cache_type=GlobalBatchsizeCache)

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
        explicit_toma_range(f, 0, 128, 64,
                            batchsize_cache_type=StacktraceMemoryBatchsizeCache)

    explicit_toma_range(f, 0, 128, 64,
                        batchsize_cache_type=StacktraceMemoryBatchsizeCache)

    assert batchsizes == [64, 64, 32, 32, 16, 16] + [16] * 8 + [64, 64, 32, 32,
                                                                16, 16]


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
        explicit_toma_range(f, 0, 128, 64,
                            batchsize_cache_type=GlobalBatchsizeCache)

    explicit_toma_range(f, 0, 128, 64,
                        batchsize_cache_type=GlobalBatchsizeCache)

    for _ in range(2):
        explicit_toma_range(f, 0, 128, 64,
                            batchsize_cache_type=StacktraceMemoryBatchsizeCache)

    assert batchsizes == ([64, 64, 32, 32, 16, 16] + [16] * 8 * 2 +
                          [64, 64, 32, 32, 16, 16] + [16] * 8)
