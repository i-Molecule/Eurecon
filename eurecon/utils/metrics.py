"""Metrics module."""
import datetime


def now():
    """Returns current datetime."""
    return datetime.datetime.now()


def timing(func):
    """Decorator for timing metrics."""

    def wrapper(*args, **kwargs):
        start_time = now()
        result = func(*args, **kwargs)
        elapsed_time = now() - start_time

        object = args[0]

        func_name = func.__name__

        if not getattr(object, "metrics", False):
            setattr(object, "metrics", {})

        if func_name not in object.metrics:
            object.metrics[func_name] = []

        object.metrics[func_name].append(elapsed_time.total_seconds())

        return result

    return wrapper


def static_timing(metric_var: dict):
    """Декоратор для подсчета затраченного времени и отправки метрик."""

    def wrapper(func):
        def wrapped(*args, **kwargs):
            start_time = now()
            result = func(*args, **kwargs)
            elapsed_time = now() - start_time

            func_name = func.__name__

            if func_name not in metric_var:
                metric_var[func_name] = []

            metric_var[func_name].append(elapsed_time.total_seconds())
            return result

        return wrapped

    return wrapper
