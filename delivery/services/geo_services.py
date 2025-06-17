import requests
import numpy as np

def get_route_geometry(start, end, profile="driving"):
    """
    Получает маршрут между двумя точками с использованием OSRM Route API.
    
    Аргументы:
      start (tuple/list): Координаты начала маршрута в виде (lat, lon).
      end (tuple/list): Координаты конца маршрута в виде (lat, lon).
      profile (str): Режим маршрутизации ("driving", "walking", "cycling").
    
    Возвращает:
      list: Список координат маршрута в формате [[lat, lon], [lat, lon], ...].
    """
    # OSRM требует координаты в формате "долгота,широта"
    coords_str = f"{start[1]},{start[0]};{end[1]},{end[0]}"
    url = f"http://router.project-osrm.org/route/v1/{profile}/{coords_str}?overview=full&geometries=geojson"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса к OSRM: {response.status_code}")
    data = response.json()
    # Извлекаем геометрию маршрута (координаты в формате GeoJSON: [lon, lat])
    route_coords = data["routes"][0]["geometry"]["coordinates"]
    # Переводим в формат [[lat, lon], ...]
    return [[coord[1], coord[0]] for coord in route_coords]

def get_distance_duration_matrix(coords : list) -> dict[list, list]:
    """
    Получает матрицу расстояний между точками через OSRM Table API.
    
    Аргументы:
      coords - список кортежей (lat, lon) для точек.
      
    Возвращает:
      distances - матрица расстояний (в метрах) между точками.
    """
    # OSRM требует, чтобы координаты были в формате "долгота,широта" и разделены точкой с запятой
    coords_str = ';'.join([f"{lon},{lat}" for lat, lon in coords])
    # Формируем URL запроса к OSRM Table API
    url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance,duration"
    print("url_OSRM", url)
    # print("url = ", url)
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Ошибка запроса к OSRM Table API.", response.status_code)
    
    data = response.json()
    # print("data: ", data.get('distances'), data.get('durations'))
    return {"distances" : data.get('distances'), "durations" : data.get('durations')}

def geocode(address):
    """
    Получает координаты (latitude, longitude) для заданного адреса с использованием Yandex Geocode API.
    
    :param address: Строка с адресом, например, "проспект Луначарского, 62к1, Санкт-Петербург"
    :return: Кортеж (latitude, longitude)
    """
    # Замените этот API-ключ на реальный, если потребуется.
    api = "02ce22ae-aaa3-400d-8ab5-1573f0cf3515"
    # Форматируем адрес для URL: заменяем пробелы знаком +
    formatted_address = "+".join(address.split())
    
    # Добавляем параметр format=json, чтобы получить ответ в формате JSON.
    url = f"https://geocode-maps.yandex.ru/1.x/?apikey={api}&geocode={formatted_address}&format=json"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Ошибка запроса к Yandex API.")
    
    data = response.json()
    
    try:
        feature_members = data["response"]["GeoObjectCollection"]["featureMember"]
        if not feature_members:
            raise Exception("Маршрут не найден.")
        
        # Извлекаем позицию из первого найденного GeoObject.
        pos = feature_members[0]["GeoObject"]["Point"]["pos"]
        # pos имеет формат "долгота широта"
        lon, lat = map(float, pos.split())
        return lat, lon
    except (KeyError, IndexError) as e:
        raise Exception("Ошибка обработки данных геокодирования: " + str(e))
    
def compute_osrm_matrices(coords):
    """
    Возвращает duration_matrix и distance_matrix (в секундах и метрах) 
    для списка координат [(lon,lat), ...] через OSRM table API.
    """
    coord_str = ";".join(f"{lon},{lat}" for lon,lat in coords)
    url = f"http://router.project-osrm.org/table/v1/driving/{coord_str}?annotations=duration,distance"
    print("url_osrm_matrix", url)
    resp = requests.get(url).json()
    return np.array(resp["durations"]), np.array(resp["distances"])

def get_osrm_trip_order(coords):
    """
    Возвращает список индексов оптимального по расстоянию маршрута через OSRM trip API.
    """
    coord_str = ";".join(f"{lon},{lat}" for lon,lat in coords)
    url = f"http://router.project-osrm.org/trip/v1/driving/{coord_str}?roundtrip=false&annotations=distance&source=first"
    print("url_get_osrm_trip_order", url)
    trip = requests.get(url).json()
    # waypoints[].waypoint_index — порядок визитов оригинальных точек
    order = [wp["waypoint_index"] for wp in sorted(trip["waypoints"], key=lambda w: w["trips_index"])]
    print("order", order)
    return order 
