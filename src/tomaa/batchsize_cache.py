import functools
from dataclasses import dataclass
from typing import Optional

from tomaa import stacktrace as tst, torch_cuda_memory as tcm
import weakref


@dataclass
class Batchsize:
    value: Optional[int] = None

    def set_initial_batchsize(self, initial_batchsize):
        if not self.value:
            self.value = initial_batchsize

    def get_batchsize(self):
        return self.value

    def decrease_batchsize(self):
        self.value //= 2


class BatchsizeCache:
    all_instances = weakref.WeakValueDictionary()

    def __init__(self):
        stacktrace = tst.get_simple_traceback(2)
        BatchsizeCache.all_instances[stacktrace] = self

    def get_batchsize(self, initial_batchsize: int) -> Batchsize:
        raise NotImplementedError()

    @staticmethod
    def get_attr_suffix() -> str:
        raise NotImplementedError()


@dataclass
class NoBatchsizeCache(BatchsizeCache):
    def get_batchsize(self, initial_batchsize) -> Batchsize:
        return Batchsize(initial_batchsize)


@dataclass
class GlobalBatchsizeCache(BatchsizeCache):
    batchsize: Optional[Batchsize] = None

    def get_batchsize(self, initial_batchsize) -> Batchsize:
        if not self.batchsize:
            self.batchsize = Batchsize(initial_batchsize)
        return self.batchsize


class StacktraceMemoryBatchsizeCache(BatchsizeCache):
    LRU_CACHE_SIZE = 128
    initial_batchsize: Optional[int]

    def __init__(self, lru_cache_size=None):
        super().__init__()

        self.initial_batchsize = None

        @functools.lru_cache(lru_cache_size or StacktraceMemoryBatchsizeCache.LRU_CACHE_SIZE)
        def get_batchsize_from_cache(stacktrace, available_memory):
            return Batchsize(self.initial_batchsize)

        self.get_batchsize_from_cache = get_batchsize_from_cache

    def get_batchsize(self, initial_batchsize):
        stacktrace = tst.get_simple_traceback(2)
        available_memory_256MB = int(tcm.get_cuda_assumed_available_memory() // 2 ** 28)

        batchsize = self.get_batchsize_from_cache(stacktrace, available_memory_256MB)
        batchsize.set_initial_batchsize(initial_batchsize)
        return batchsize
