# Torch Memory-adaptive Algorithms (TOMA)

[![Build Status](https://www.travis-ci.com/BlackHC/toma.svg?branch=master)](https://www.travis-ci.com/BlackHC/toma) [![codecov](https://codecov.io/gh/BlackHC/toma/branch/master/graph/badge.svg)](https://codecov.io/gh/BlackHC/toma) [![PyPI](https://img.shields.io/badge/PyPI-toma-blue.svg)](https://pypi.python.org/pypi/toma/)

A collection of helpers to make it easier to write code that adapts to the available (CUDA) memory.
Specifically, it retries code that fails due to OOM (out-of-memory) conditions and lowers batchsizes automatically. 

To avoid failing over repeatedly, a simple cache is implemented that memorizes that last successful batchsize given the call and available free memory.

## Installation

To install using pip, use:

```
pip install toma
```

To run the tests, use:

```
python setup.py test
```

## Example

```python
from toma import toma

@toma.batch(initial_batchsize=512)
def run_inference(batchsize, model, dataset):
    # ...

run_inference(batchsize, model, dataset)
```

This will try to execute train_model with batchsize=512. If a memory error is thrown, it will decrease the batchsize until it succeeds.

**Note:** 
This batch size can be different from the batch size used to accumulate gradients by only calling `optimizer.step()` every so often.

To make it easier to loop over a ranges, there are also `toma.range` and `toma.chunked`:

```python
@toma.chunked(initial_step=512)
def compute_result(out: torch.Tensor, start: int, end: int):
    # ...

result = torch.empty((8192, ...))
compute_result(result)
```

This will chunk `result` and pass the chunks to `compute_result` one by one. 
Again, if it fails due to OOM, the step will be halfed etc.
Compared to `toma.batch`, this allows for reduction of the step size while looping over the chunks.
This can save computation.

```python
@toma.range(initial_step=32)
def reduce_data(start: int, end: int, out: torch.Tensor, dataA: torch.Tensor, dataB: torch.Tensor):
    # ...

reduce_data(0, 1024, result, dataA, dataB)
``` 

`toma.range` iterates over `range(start, end, step)` with `step=initial_step`. If it fails due to OOM, it will lower the step size and continue.

### `toma.execute`

To make it easier to just execute a block without having to extract it into a function and then call it, we also provide `toma.execute.batch`, `toma.execute.range` and `toma.execute.chunked`, which are somewhat unorthodox and call the function that is passed to them right away. (Mainly because there is no support for anonymous functions in Python beyond lambda expressions.)

```python
def function():
    # ... other code

    @toma.execute.chunked(batched_data, initial_step=128):
    def compute(chunk, start, end):
        # ...
```

## Cache

There are 3 available cache types at the moment. 
They can be changed by either setting `toma.DEFAULT_CACHE_TYPE` or by passing `cache_type` to the calls.

For example:
```python
@toma.batch(initial_batchsize=512, cache_type=toma.GlobalBatchsizeCache)
```
or
```python
toma.explicit.batch(..., toma_cache_type=toma.GlobalBatchsizeCache)
```

### `StacktraceMemoryBatchsizeCache`: Stacktrace & Available Memory (*the default*)

This memorizes the successful batchsizes for a given call trace and available memory at that point.
For most machine learning code, this is sufficient to remember the right batchsize without having to look at the actual arguments and understanding more of the semantics.

The implicit assumption is that after a few iterations a stable state will be reached in regards to GPU and CPU memory usage.

To limit the CPU memory of the process, toma provides:
```python
import toma.cpu_memory

toma.cpu_memory.set_cpu_memory_limit(8)
```
This can also be useful to avoid accidental swap thrashing.

### `GlobalBatchsizeCache`: Global per Function

This reuses the last successful batchsize independently from where the call happened.

### `NoBatchsizeCache`: No Caching

Always starts with the suggested batchsize and fails over if necessary.

## Benchmark/Overhead

There is overhead involved. Toma should only be used with otherwise time/memory-consuming operations.

```text
---------------------------------------------------------------------------------- benchmark: 5 tests ----------------------------------------------------------------------------------
Name (time in ms)          Min                Max               Mean            StdDev             Median                IQR            Outliers       OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_native             2.1455 (1.0)       3.7733 (1.0)       2.3037 (1.0)      0.1103 (1.0)       2.2935 (1.0)       0.1302 (1.0)          81;5  434.0822 (1.0)         448           1
test_simple            17.4657 (8.14)     27.0049 (7.16)     21.0453 (9.14)     2.6233 (23.79)    20.4881 (8.93)      3.4384 (26.42)        13;0   47.5165 (0.11)         39           1
test_toma_no_cache     31.4380 (14.65)    40.8567 (10.83)    33.2749 (14.44)    2.2530 (20.43)    32.2698 (14.07)     2.8210 (21.67)         4;1   30.0527 (0.07)         25           1
test_explicit          33.0759 (15.42)    52.1866 (13.83)    39.6956 (17.23)    6.9620 (63.14)    38.4929 (16.78)    11.2344 (86.31)         4;0   25.1917 (0.06)         20           1
test_toma              36.9633 (17.23)    57.0220 (15.11)    43.5201 (18.89)    6.7318 (61.05)    41.6034 (18.14)     7.2173 (55.45)         2;2   22.9779 (0.05)         13           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

## Thanks

Thanks to [@y0ast](https://github.com/y0ast) for feedback and discussion.
