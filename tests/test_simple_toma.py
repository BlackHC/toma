import torch

from toma import simple


def raise_fake_oom():
    raise RuntimeError("CUDA out of memory.")


def test_fake_simple_toma():
    hit_16 = False

    def f(batch_size):
        if batch_size != 16:
            raise_fake_oom()
        if batch_size == 16:
            nonlocal hit_16
            hit_16 = True

        assert batch_size >= 16

    simple.batch(f, 64)

    assert hit_16


def test_fake_simple_toma_range():
    hit_16 = False

    def f(start, end):
        batch_size = end - start

        if batch_size != 16:
            raise_fake_oom()
        if batch_size == 16:
            nonlocal hit_16
            hit_16 = True

        assert batch_size >= 16

    simple.range(f, 0, 128, 64)

    assert hit_16


def test_fake_simple_toma_chunked():
    hit_16 = False

    def f(tensor, start, end):
        batch_size = end - start

        if batch_size != 16:
            raise_fake_oom()
        if batch_size == 16:
            nonlocal hit_16
            hit_16 = True

        tensor[:] = 1

    tensor = torch.zeros(128, dtype=torch.float)
    simple.chunked(f, tensor, 64)
    assert torch.allclose(tensor, torch.tensor(1.0))
    assert hit_16


def test_simple_toma():
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    failed = False
    succeeded = False

    def f(batch_size):
        # 2**20*2*7*2**3 * batch_size = batch_size GB
        try:
            torch.empty((batch_size, 1024, 1024, 128), dtype=torch.double, device="cuda")
        except:
            nonlocal failed
            failed = True

        nonlocal succeeded
        succeeded = True

    simple.batch(f, 64)
    assert failed
    assert succeeded


def test_simple_toma_range():
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    failed = False
    succeeded = False

    def f(start, end):
        # 2**20*2*7*2**3 * batch_size = batch_size GB
        try:
            torch.empty((end - start, 1024, 1024, 128), dtype=torch.double, device="cuda")
        except:
            nonlocal failed
            failed = True

        nonlocal succeeded
        succeeded = True

    simple.range(f, 0, 128, 64)
    assert failed
    assert succeeded
