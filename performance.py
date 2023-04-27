from time import perf_counter

__all__ = ["track_class"]


def track_func(func):
    def _track_func(*args, **kwargs):
        _track_func.calls += 1
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        _track_func.times.append(t2 - t1)
        return result
    _track_func.calls = 0
    _track_func.times = []
    return _track_func


def track_class(cls):
    for func in cls.__dict__.values():
        if hasattr(func, "__call__") and func.__name__ != "__init__":
            setattr(cls, func.__name__, track_func(func))
    setattr(cls, "print_results", print_results)
    return cls


def print_results(obj):
    for name in dir(obj):
        func = getattr(obj, name)
        if hasattr(func, "__call__") and hasattr(func, "calls"):
            times = func.times
            calls = func.calls
            if calls == 0:
                continue

            avg = sum(times)/len(times)
            name = name[:20] + " " * (20 - len(name))
            print(f"{name} {calls}-calls\t avg {avg:.2E}s")
