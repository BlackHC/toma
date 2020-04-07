from tomma import stacktrace


def get_stacktrace():
    return stacktrace.get_simple_traceback()


def outer_func():
    return get_stacktrace()


def test_get_simple_traceback():
    stacktrace1 = outer_func()
    stacktrace2 = outer_func()

    assert hash(stacktrace1) != hash(stacktrace2)
    assert stacktrace1 != stacktrace2

    stacktraces = []
    for i in range(2):
        stacktraces.append(outer_func())

    assert stacktraces[0] == stacktraces[1]
    assert hash(stacktraces[0]) == hash(stacktraces[1])



def test_watermark():
    stacktrace1 = outer_func()

    assert len(stacktrace1) > 1

    with stacktrace.watermark():
        stacktrace2 = outer_func()

    assert len(stacktrace2) == 3

    stacktrace3 = outer_func()
    assert len(stacktrace1) == len(stacktrace3)


def test_set_watermark():
    @stacktrace.set_watermark
    def outer_func2():
        return outer_func()

    assert len(outer_func2()) == 3