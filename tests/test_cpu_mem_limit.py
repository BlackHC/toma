import torch

from toma import toma
from toma import cpu_memory


def test_cpu_mem_limit():
    cpu_memory.set_cpu_memory_limit(2)

    batchsize = None

    @toma.batch(initial_batchsize=2048)
    def allocate_gigabytes(bs):
        torch.empty((bs, 1024, 1024 // 4), dtype=torch.float32)

        nonlocal batchsize
        batchsize = bs

    allocate_gigabytes()

    assert batchsize <= 512
