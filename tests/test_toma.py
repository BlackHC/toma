from tomaa import toma, toma_range, toma_chunked
from tomaa import simple, simple_range
from tomaa import explicit_toma, explicit_toma_range


def raise_fake_oom():
    raise RuntimeError("CUDA out of memory.")


def test_fake_toma_simple():
    batchsizes = []

    @toma(method=simple)
    def f(batchsize):
        nonlocal batchsizes
        batchsizes.append(batchsize)

        if batchsize != 16:
            raise_fake_oom()

    for _ in range(2):
        f(64)

    assert batchsizes == [64, 32, 16, 64, 32, 16]


def test_fake_toma_explicit():
    batchsizes = []

    @toma(method=explicit_toma)
    def f(batchsize):
        nonlocal batchsizes
        batchsizes.append(batchsize)

        if batchsize != 16:
            raise_fake_oom()

    for _ in range(2):
        f(64)

    assert batchsizes == [64, 32, 16, 16]


def test_fake_toma_range_global():
    batchsizes = []

    @toma_range(method=simple_range)
    def f(start, end):
        batchsize = end - start

        nonlocal batchsizes
        batchsizes.append(batchsize)

        remaining = 128 - end

        if batchsize > 16 and batchsize > remaining:
            raise_fake_oom()

    for _ in range(2):
        f(0, 128, maa_initial_step=64)

    assert batchsizes == [64, 64, 32, 32, 16, 16] * 2


def test_fake_toma_range_explicit():
    batchsizes = []

    @toma_range(method=explicit_toma_range)
    def f(start, end):
        batchsize = end - start

        nonlocal batchsizes
        batchsizes.append(batchsize)

        remaining = 128 - end

        if batchsize > 16 and batchsize > remaining:
            raise_fake_oom()

    for _ in range(2):
        f(0, 128, maa_initial_step=64)

    assert batchsizes == [64, 64, 32, 32, 16, 16] + [16] * 8