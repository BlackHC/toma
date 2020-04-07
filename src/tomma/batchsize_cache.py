import functools
from dataclasses import dataclass
from typing import Optional

from tomma import stacktrace as tst, torch_cuda_memory as tcm
import weakref


@dataclass
class Batchsize:
    value: int

    def get_batchsize(self):
        return self.value

    def decrease_batchsize(self):
        self.value //= 2


class BatchsizeCache:
    all_instances = weakref.WeakValueDictionary()

    def __init__(self):
        stacktrace = tst.get_simple_traceback(2)
        BatchsizeCache.all_instances[stacktrace] = self

    def set_initial_batchsize(self, initial_batchsize):
        raise NotImplementedError()

    def get_batchsize(self) -> Batchsize:
        raise NotImplementedError()


@dataclass
class GlobalBatchsizeCache(BatchsizeCache):
    value: Optional[Batchsize]

    def set_initial_batchsize(self, initial_batchsize):
        if not self.value:
            self.value = Batchsize(initial_batchsize)

    def get_batchsize(self) -> Batchsize:
        return self.value


@dataclass
class StacktraceMemoryBatchsizeCache(BatchsizeCache):
    initial_batchsize: int

    def set_initial_batchsize(self, initial_batchsize):
        self.initial_batchsize = initial_batchsize

    @functools.lru_cache
    def get_batchsize_from_cache(self, stacktrace, available_memory):
        return Batchsize(self.initial_batchsize)

    def get_batchsize(self):
        stacktrace = tst.get_simple_traceback(2)
        available_memory_256MB = int(tcm.get_cuda_assumed_available_memory() // 2 ** 28)

        return self.get_batchsize_from_cache(stacktrace, available_memory_256MB)
