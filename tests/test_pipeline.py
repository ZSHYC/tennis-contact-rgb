from threading import Event

import pytest


def test_prefetch_overlaps_one_next_item_and_keeps_order():
    from tennis_contact.pipeline import prefetch

    second_started = Event()
    release = Event()
    loaded = []

    def load(value):
        loaded.append(value)
        if value == 1:
            second_started.set()
            assert release.wait(5)
        return value * 2

    try:
        with prefetch(range(4), load) as results:
            assert next(results) == 0
            assert second_started.wait(5)
            assert loaded == [0, 1]
            release.set()
            assert list(results) == [2, 4, 6]
    finally:
        release.set()


def test_prefetch_propagates_read_errors_and_stops_on_early_exit():
    from tennis_contact.pipeline import prefetch

    loaded = []

    def load(value):
        loaded.append(value)
        if value == 2:
            raise ValueError("decode failed")
        return value

    with prefetch(range(5), load) as results:
        with pytest.raises(ValueError, match="decode failed"):
            list(results)
    assert loaded == [0, 1, 2]
    loaded.clear()
    with prefetch(range(5), load) as results:
        assert next(results) == 0
    assert loaded in ([0], [0, 1])
    with prefetch([], load) as results:
        assert list(results) == []
