import pytest
import torch

from toma import cpu_memory


@pytest.mark.forked
def test_cpu_mem_limit():
    tensor = torch.empty((128, 1024, 1024 // 4), dtype=torch.float32)
    tensor.resize_(1)

    cpu_memory.set_cpu_memory_limit(0.25)

    with pytest.raises(RuntimeError):
        torch.empty((512, 1024, 1024 // 4), dtype=torch.float32)
