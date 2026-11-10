"""在计算当前窗口时读取下一个窗口，限制在一个待取结果。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def prefetch(items, load):
    items = iter(items)
    pending = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-read") as executor:
        def results():
            nonlocal pending
            for item in items:
                pending = executor.submit(load, item)
                break
            while pending is not None:
                value = pending.result()
                pending = None
                for item in items:
                    pending = executor.submit(load, item)
                    break
                yield value

        iterator = results()
        try:
            yield iterator
        finally:
            iterator.close()
            if pending is not None:
                pending.cancel()
