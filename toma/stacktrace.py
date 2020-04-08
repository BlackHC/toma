import functools
import inspect
from contextlib import contextmanager

__watermark = 0


def _constant_code_context(code_context):
    if not code_context:
        return None
    if len(code_context) == 1:
        return code_context[0]
    return tuple(code_context)


def get_simple_traceback(ignore_top=0):
    """Get a simple trackback that can be hashed and won't create reference
    cyles."""
    stack = inspect.stack(context=1)[ignore_top + 1 : -__watermark - 1]
    simple_traceback = tuple(
        (fi.filename, fi.lineno, fi.function, _constant_code_context(fi.code_context), fi.index) for fi in stack
    )
    return simple_traceback


@contextmanager
def watermark():
    global __watermark
    old_watermark = __watermark

    # Remove the entries for `watermark` and
    # `contextmanager.__enter__` and for the with block.
    # Remove another one to keep the caller.
    __watermark = len(inspect.stack(context=0)) - 4

    try:
        yield
    finally:
        __watermark = old_watermark


def set_watermark(func):
    @functools.wraps(func)
    def watermark_wrapper(*args, **kwargs):
        global __watermark
        old_watermark = __watermark

        # Remove the entries for `watermark` and
        # `contextmanager.__enter__`.
        # Dump frames for this wrapper.
        __watermark = len(inspect.stack(context=0)) - 1

        try:
            return func(*args, **kwargs)
        finally:
            __watermark = old_watermark

    return watermark_wrapper
