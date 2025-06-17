from typing import Dict, Any
from delivery.models import Courier, Order
import logging
import pandas as pd
import folium
from folium.plugins import PolyLineTextPath
import time
from .geo_services import get_route_geometry, geocode, get_distance_duration_matrix


logger = logging.getLogger(__name__)

# def get_route_geometry(start, end, profile="driving"):
#     """
#     Получает маршрут между двумя точками с использованием OSRM Route API.
    
#     Аргументы:
#       start (tuple/list): Координаты начала маршрута в виде (lat, lon).
#       end (tuple/list): Координаты конца маршрута в виде (lat, lon).
#       profile (str): Режим маршрутизации ("driving", "walking", "cycling").
    
#     Возвращает:
#       list: Список координат маршрута в формате [[lat, lon], [lat, lon], ...].
#     """
#     # OSRM требует координаты в формате "долгота,широта"
#     coords_str = f"{start[1]},{start[0]};{end[1]},{end[0]}"
#     url = f"http://router.project-osrm.org/route/v1/{profile}/{coords_str}?overview=full&geometries=geojson"
#     response = requests.get(url)
#     if response.status_code != 200:
#         raise Exception(f"Ошибка запроса к OSRM: {response.status_code}")
#     data = response.json()
#     # Извлекаем геометрию маршрута (координаты в формате GeoJSON: [lon, lat])
#     route_coords = data["routes"][0]["geometry"]["coordinates"]
#     # Переводим в формат [[lat, lon], ...]
#     return [[coord[1], coord[0]] for coord in route_coords]

# def get_distance_duration_matrix(coords : list) -> dict[list, list]:
#     """
#     Получает матрицу расстояний между точками через OSRM Table API.
    
#     Аргументы:
#       coords - список кортежей (lat, lon) для точек.
      
#     Возвращает:
#       distances - матрица расстояний (в метрах) между точками.
#     """
#     # OSRM требует, чтобы координаты были в формате "долгота,широта" и разделены точкой с запятой
#     coords_str = ';'.join([f"{lon},{lat}" for lat, lon in coords])
#     # Формируем URL запроса к OSRM Table API
#     url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance,duration"
#     print("url_OSRM", url)
#     # print("url = ", url)
#     response = requests.get(url)
#     if response.status_code != 200:
#         raise Exception("Ошибка запроса к OSRM Table API.", response.status_code)
    
#     data = response.json()
#     # print("data: ", data.get('distances'), data.get('durations'))
#     return {"distances" : data.get('distances'), "durations" : data.get('durations')}

# def geocode(address):
#     """
#     Получает координаты (latitude, longitude) для заданного адреса с использованием Yandex Geocode API.
    
#     :param address: Строка с адресом, например, "проспект Луначарского, 62к1, Санкт-Петербург"
#     :return: Кортеж (latitude, longitude)
#     """
#     # Замените этот API-ключ на реальный, если потребуется.
#     api = "02ce22ae-aaa3-400d-8ab5-1573f0cf3515"
#     # Форматируем адрес для URL: заменяем пробелы знаком +
#     formatted_address = "+".join(address.split())
    
#     # Добавляем параметр format=json, чтобы получить ответ в формате JSON.
#     url = f"https://geocode-maps.yandex.ru/1.x/?apikey={api}&geocode={formatted_address}&format=json"
    
#     response = requests.get(url)
#     if response.status_code != 200:
#         raise Exception("Ошибка запроса к Yandex API.")
    
#     data = response.json()
    
#     try:
#         feature_members = data["response"]["GeoObjectCollection"]["featureMember"]
#         if not feature_members:
#             raise Exception("Маршрут не найден.")
        
#         # Извлекаем позицию из первого найденного GeoObject.
#         pos = feature_members[0]["GeoObject"]["Point"]["pos"]
#         # pos имеет формат "долгота широта"
#         lon, lat = map(float, pos.split())
#         return lat, lon
#     except (KeyError, IndexError) as e:
#         raise Exception("Ошибка обработки данных геокодирования: " + str(e))

    
# def get_route_geometry(start, end, profile="driving"):
#     """
#     Получает маршрут между двумя точками с использованием OSRM Route API.
    
#     Аргументы:
#       start (tuple/list): Координаты начала маршрута в виде (lat, lon).
#       end (tuple/list): Координаты конца маршрута в виде (lat, lon).
#       profile (str): Режим маршрутизации ("driving", "walking", "cycling").
    
#     Возвращает:
#       list: Список координат маршрута в формате [[lat, lon], [lat, lon], ...].
#     """
#     # OSRM требует координаты в формате "долгота,широта"
#     coords_str = f"{start[1]},{start[0]};{end[1]},{end[0]}"
#     url = f"http://router.project-osrm.org/route/v1/{profile}/{coords_str}?overview=full&geometries=geojson"
#     response = requests.get(url)
#     if response.status_code != 200:
#         raise Exception(f"Ошибка запроса к OSRM: {response.status_code}")
#     data = response.json()
#     # Извлекаем геометрию маршрута (координаты в формате GeoJSON: [lon, lat])
#     route_coords = data["routes"][0]["geometry"]["coordinates"]
#     # Переводим в формат [[lat, lon], ...]
#     return [[coord[1], coord[0]] for coord in route_coords]



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

# def geocluster(data, n_clusters, ALPHA,BETA, vcw, vcv):
#     orders = [Order.objects.get(order_number=rec[0]) for rec in data]
#     info = cluster_orders(orders, n_clusters=n_clusters, alpha=ALPHA, beta=BETA, capacity_volume=vcv, capacity_weight=vcw)
#     labels = info.get('labels', [])+1
#     print("geocluster_labels", labels)
#     return {order.order_number: label for order, label in zip(orders, labels)}


# def optimize_route(points, capacity_weight, capacity_volume, ALPHA=0, BETA=10000):
#     orders_df = orders_to_dataframe()

#     # Отфильтруем DataFrame по точкам данного кластера

#     order_ids = [rec[0] for rec in points]
#     cluster_df = orders_df.loc[order_ids]

#     # Список объектов-строк (Series) с корректным .name (order_number)
#     orders_cluster = [cluster_df.loc[oid] for oid in order_ids]

#     route = solve_vrptw(
#         orders_cluster,
#         orders_df,
#         vehicle_capacity_weight=capacity_weight,
#         vehicle_capacity_volume=capacity_volume, ALPHA=ALPHA, BETA=BETA
#     )


#     return route

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