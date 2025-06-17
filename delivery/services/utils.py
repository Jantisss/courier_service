from typing import Dict, Any
from delivery.models import Courier, Order
import logging
import pandas as pd
import folium
from folium.plugins import PolyLineTextPath
import time
from .geo_services import get_route_geometry, geocode, get_distance_duration_matrix


logger = logging.getLogger(__name__)

def get_route(route_coords):
    len_c = len(route_coords)
    pairs_coords = []
    for i in range(len_c):
        if i == len_c-1:
            break
        pairs_coords.append((route_coords[i], route_coords[i+1]))

    geometry_pol = []
    for pair_coords in pairs_coords:
        geometry_pol.append(get_route_geometry(pair_coords[0],pair_coords[1]))

    return {"pairs_coords" : pairs_coords,"geometry_pol" : geometry_pol}

def flatten(x):
    if isinstance(x, tuple):
        items = []
        for item in x:
            items.extend(flatten(item))
        return items
    else:
        return [x]
    

    
def add_route_with_labels(route_poly_coords, cluster, cluster_to_color, map_obj, route_coords):
    """
    Добавляет на карту маршрут с указанием порядковых номеров точек и стрелками, показывающими направление.
    Аргументы:
      route_coords (list): Список координат маршрута, где каждая координата имеет вид (lat, lon).
      cluster (int или str): Номер кластера, для подписи.
      cluster_to_color (dict): Словарь с цветами для каждого кластера.
      map_obj: Объект folium.Map.
    """
    # Добавляем полилинию маршрута
    polyline = folium.PolyLine(route_poly_coords, color=cluster_to_color.get(cluster, 'gray'), weight=3)
    polyline.add_to(map_obj)
    
    # Добавляем стрелки вдоль маршрута для указания направления
    PolyLineTextPath(
        polyline, 
        '   ▶   ',         # текст для отображения (символ стрелки)
        repeat=True,        # повторяем стрелку по всей длине
        offset=7,           # смещение стрелки от линии
        attributes={'font-size': '14', 'fill': cluster_to_color.get(cluster, 'gray')}
    ).add_to(map_obj)
    
    
    # Добавляем маркировку точек (с номерами) вдоль маршрута
    for i, coord in enumerate(route_coords):
        folium.Marker(
            location=coord,
            icon=folium.DivIcon(
                html=f'<div style="font-size: 12pt; color: black; pointer-events: none;"">{i+1}</div>',
                icon_anchor=(0, -8)
            )
        ).add_to(map_obj)

def orders_to_dataframe() -> pd.DataFrame:
    """
    Преобразует все Order в pandas.DataFrame для дальнейшей обработки.
    Индексом будет order_number.
    """
    qs = Order.objects.all().values(
        'order_number', 'address', 'lat', 'lon',
        'delivery_start', 'delivery_end', 'weight', 'volume', 'comment', 'cluster'
    )
    df = pd.DataFrame.from_records(qs)
    df['lat'] = df['lat'].astype(float)
    df['lon'] = df['lon'].astype(float)
    return df.set_index('order_number')

def time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute

def assign_couriers_to_clusters(clusters_map: Dict[Any, int]) -> Dict[int, Courier]:
    """
    Определяет, какому курьеру назначить каждый кластер.

    Если курьеров меньше, чем кластеров, лишние кластеры останутся без назначения.

    :param clusters_map: Словарь {order_id: cluster_index}
    :return: Словарь {cluster_index: Courier instance}
    """
    # Получаем список доступных курьеров
    available = list(Courier.objects.filter(is_available=True))
    if not available:
        logger.warning('Нет доступных курьеров для назначения')
        return {}

    # Находим уникальные кластеры
    clusters = sorted(set(clusters_map.values()))
    n_couriers = len(available)
    n_clusters = len(clusters)

    cluster_to_courier: Dict[int, Courier] = {}
    # Распределяем по возможности
    for i, cluster in enumerate(clusters):
        if i < n_couriers:
            courier = available[i]
            cluster_to_courier[cluster] = courier
        else:
            # Курьеров не хватает, оставляем кластер без назначения
            logger.warning(f'Кластер {cluster} не назначен: недостаточно курьеров')
            break

    return cluster_to_courier
