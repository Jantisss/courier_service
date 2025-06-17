import datetime
import pytest
import numpy as np

from delivery.services import clustering_solveVRPTW as csvr


class DummyOrder:
    def __init__(self, lat, lon, delivery_start):
        self.lat = lat
        self.lon = lon
        self.delivery_start = delivery_start


def test_temporal_penalty_same_block():
    o1 = DummyOrder(0, 0, datetime.time(hour=8, minute=30))
    o2 = DummyOrder(0, 0, datetime.time(hour=9, minute=15))
    # оба в блоке 8//2=4 и 9//2=4 → штраф 0
    assert csvr.temporal_penalty(o1, o2) == 0


def test_temporal_penalty_diff_block():
    o1 = DummyOrder(0, 0, datetime.time(hour=8))
    o2 = DummyOrder(0, 0, datetime.time(hour=12))
    # блоки 8//2=4, 12//2=6 → |4−6|*2=4
    assert csvr.temporal_penalty(o1, o2) == 4


def test_flatten_route_element():
    # одиночное число
    assert csvr.flatten_route_element(5) == [5]
    # вложённые списки
    assert csvr.flatten_route_element([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
    # другой тип → пустой результат
    assert csvr.flatten_route_element("abc") == []




def test_perform_clustering_simple():
    # две близкие точки и одна далёкая → шум
    class P:
        def __init__(self, lat, lon):
            self.lat = lat
            self.lon = lon
            # чтобы temporal_penalty не падал
            self.delivery_start = datetime.time(hour=0)

    orders = [P(0, 0), P(0, 0.1), P(10, 10)]
    labels = csvr.perform_clustering(orders, eps=0.5, min_samples=2)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (3,)
    # первые две должны быть в одном кластере, третья — шум (-1)
    assert labels[0] == labels[1]
    assert labels[2] == -1


