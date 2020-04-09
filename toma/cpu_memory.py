import psutil


def get_available_cpu_memory():
    this_process = psutil.Process()
    available_memory = psutil.virtual_memory().available

    try:
        import resource

        soft_mem_limit, hard_mem_limit = resource.getrlimit(resource.RLIMIT_AS)
        if hard_mem_limit != resource.RLIM_INFINITY:
            used_memory = this_process.memory_info().vms
            available_memory = min(hard_mem_limit - used_memory, available_memory)
    except ImportError:
        pass

    return available_memory


def set_cpu_memory_limit(num_gigabytes):
    try:
        import resource

        num_bytes = int(num_gigabytes * 2 ** 30)
        _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
        if hard_limit != resource.RLIM_INFINITY:
            hard_limit = min(num_bytes, hard_limit)
        else:
            hard_limit = num_bytes
        resource.setrlimit(resource.RLIMIT_AS, (hard_limit, hard_limit))
    except ImportError:
        pass


def is_out_of_cpu_memory(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )
