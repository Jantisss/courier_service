import datetime
import logging
import pandas as pd
import pytest

import delivery.services.utils as utils
from delivery.models import Order, Courier
from django.contrib.auth.models import User

# Для тестирования картографических функций
# будем подменять Folium и PolyLineTextPath

def test_get_route(monkeypatch):
    # Подменяем get_route_geometry
    captured = {'calls': []}
    def fake_route_geom(a, b):
        captured['calls'].append((a, b))
        return {'type': 'LineString', 'coordinates': [a, b]}
    monkeypatch.setattr(utils, 'get_route_geometry', fake_route_geom)

    coords = [(0, 0), (1, 1), (2, 2)]
    result = utils.get_route(coords)

    # Проверяем пары координат
    assert result['pairs_coords'] == [((0, 0), (1, 1)), ((1, 1), (2, 2))]
    # Проверяем вызовы геометрии
    assert captured['calls'] == [((0, 0), (1, 1)), ((1, 1), (2, 2))]
    # Проверяем структуру geometry_pol
    for geom in result['geometry_pol']:
        assert geom['type'] == 'LineString'


def test_flatten():
    # Одиночное значение
    assert utils.flatten(5) == [5]
    # Вложенные кортежи
    assert utils.flatten((1, (2, 3), 4)) == [1, 2, 3, 4]
    # Не-кортеж
    assert utils.flatten("abc") == ["abc"]


def test_add_route_with_labels(monkeypatch):
    import delivery.services.utils as u
    calls = {}

    # Fake PolyLine
    class FakePolyline:
        def __init__(self, coords, color, weight):
            calls['polyline'] = {'coords': coords, 'color': color, 'weight': weight}
        def add_to(self, m):
            calls.setdefault('polyline_to', []).append(m)

    # Fake PolyLineTextPath
    class FakeTextPath:
        def __init__(self, polyline, text, repeat, offset, attributes):
            calls['textpath'] = {'polyline': polyline, 'text': text, 'repeat': repeat, 'offset': offset, 'attributes': attributes}
        def add_to(self, m):
            calls.setdefault('textpath_to', []).append(m)

    # Fake DivIcon
    class FakeDivIcon:
        def __init__(self, html, icon_anchor):
            calls.setdefault('divicon', []).append({'html': html, 'icon_anchor': icon_anchor})

    # Fake Marker
    class FakeMarker:
        def __init__(self, location, icon):
            calls.setdefault('marker', []).append({'location': location, 'icon': icon})
        def add_to(self, m):
            calls.setdefault('marker_to', []).append(m)

    # Подменяем
    monkeypatch.setattr(u, 'PolyLineTextPath', FakeTextPath)
    monkeypatch.setattr(u.folium, 'PolyLine', FakePolyline)
    monkeypatch.setattr(u.folium, 'Marker', FakeMarker)
    monkeypatch.setattr(u.folium, 'DivIcon', FakeDivIcon)

    # Данные
    route_poly = [(0, 0), (1, 1)]
    route_pts = [(0, 0), (1, 1)]
    color_map = {10: 'blue'}
    mobj = object()

    # Вызов
    u.add_route_with_labels(route_poly, 10, color_map, mobj, route_pts)

    # Проверяем полилинию
    assert calls['polyline']['coords'] == route_poly
    assert calls['polyline']['color'] == 'blue'
    assert calls['polyline']['weight'] == 3
    assert mobj in calls['polyline_to']

    # Проверяем стрелки
    tp = calls['textpath']
    assert tp['repeat'] is True
    assert tp['offset'] == 7
    assert '▶' in tp['text']
    assert tp['attributes']['fill'] == 'blue'
    assert mobj in calls['textpath_to']

    # Проверяем маркеры
    assert len(calls['marker']) == len(route_pts)
    for idx, rec in enumerate(calls['marker']):
        assert rec['location'] == route_pts[idx]
    assert len(calls['divicon']) == len(route_pts)
    for d in calls['divicon']:
        assert 'pointer-events' in d['html']

@pytest.mark.django_db

def test_orders_to_dataframe():
    from datetime import time
    # Создаём несколько заказов
    o1 = Order.objects.create(
        address='A', lat=10.0, lon=20.0,
        delivery_start=time(9, 0), delivery_end=time(12, 0),
        weight=1.5, volume=0.2, contact='c1', comment='x', cluster=1
    )
    o2 = Order.objects.create(
        address='B', lat=30.0, lon=40.0,
        delivery_start=time(13, 0), delivery_end=time(16, 0),
        weight=2.0, volume=0.3, contact='c2', comment='y', cluster=2
    )

    df = utils.orders_to_dataframe()
    # Проверяем размер
    assert df.shape[0] >= 2
    # Индекс
    assert o1.order_number in df.index
    assert o2.order_number in df.index
    # Типы
    assert pd.api.types.is_float_dtype(df['lat'])
    assert pd.api.types.is_float_dtype(df['lon'])


def test_time_to_minutes():
    t = datetime.time(hour=2, minute=30)
    assert utils.time_to_minutes(t) == 150


@pytest.mark.django_db
def test_assign_couriers_to_clusters_empty(caplog):
    caplog.set_level(logging.WARNING, logger=utils.logger.name)
    # Удаляем всех доступных курьеров
    Courier.objects.all().update(is_available=False)

    res = utils.assign_couriers_to_clusters({1: 0, 2: 1})
    assert res == {}
    assert 'Нет доступных курьеров' in caplog.text


@pytest.mark.django_db
def test_assign_couriers_to_clusters_partial(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=utils.logger.name)
    # Создаём пользователей и курьеров
    u1 = User.objects.create_user(username='a')
    u2 = User.objects.create_user(username='b')
    c1 = Courier.objects.create(user=u1, login='a', fio='A', contact='1', vehicle_capacity_weight=10, vehicle_capacity_volume=0.5)
    c2 = Courier.objects.create(user=u2, login='b', fio='B', contact='2', vehicle_capacity_weight=10, vehicle_capacity_volume=0.5)

    clusters_map = {101: 0, 102: 1, 103: 2}
    res = utils.assign_couriers_to_clusters(clusters_map)

    # Должны назначить первые два кластера
    assert res[0] == c1
    assert res[1] == c2
    # Третий кластер не назначен и должен логироваться
    assert 'Кластер 2 не назначен' in caplog.text
