import numpy as np
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
# from math import floor
import requests
import json
from ..models import Order, Courier
# from .serializers import OrderSerializer, CourierSerializer
import datetime
# import pytz
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
# import pandas as pd
import requests
import numpy as np
from sklearn_extra.cluster import KMedoids
import time
from .geo_services import get_osrm_trip_order, compute_osrm_matrices
from .utils import orders_to_dataframe, add_route_with_labels, geocode, get_distance_duration_matrix, get_route_geometry, time_to_minutes

# from folium.plugins import PolyLineTextPath

def geocluster(data, n_clusters, ALPHA,BETA, vcw, vcv):
    orders = [Order.objects.get(order_number=rec[0]) for rec in data]
    info = cluster_orders(orders, n_clusters=n_clusters, alpha=ALPHA, beta=BETA, capacity_volume=vcv, capacity_weight=vcw)
    labels = info.get('labels', [])+1
    print("geocluster_labels", labels)
    return {order.order_number: label for order, label in zip(orders, labels)}


def optimize_route(points, capacity_weight, capacity_volume, ALPHA=0, BETA=10000):
    orders_df = orders_to_dataframe()

    # Отфильтруем DataFrame по точкам данного кластера

    order_ids = [rec[0] for rec in points]
    cluster_df = orders_df.loc[order_ids]

    # Список объектов-строк (Series) с корректным .name (order_number)
    orders_cluster = [cluster_df.loc[oid] for oid in order_ids]

    route = solve_vrptw(
        orders_cluster,
        orders_df,
        vehicle_capacity_weight=capacity_weight,
        vehicle_capacity_volume=capacity_volume, ALPHA=ALPHA, BETA=BETA
    )


    return route

def perform_clustering(orders, eps=0.5, min_samples=3):
    """
    Принимает список заказов и проводит кластеризацию по координатам.
    Для каждого заказа получает координаты через функцию geocode.
    """
    coordinates = []
    for order in orders:
        # Получаем координаты (lat, lng) по адресу заказа.
        lat, lng = order.lat, order.lon
        coordinates.append([lat, lng])
    coordinates = np.array(coordinates)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = db.fit_predict(coordinates)
    print(clusters)
    return clusters

# def get_distance_matrix(points, start_time = datetime.now(pytz.timezone('UTC')).isoformat(), base = {'lat' : 59.793897,'lon' : 30.409896}):
    """
    Отправляет POST-запрос к API 2GIS для получения матрицы расстояний.
    
    :param api_key: Ваш API-ключ для доступа к сервису.
    :param points: Список словарей с координатами, например:
                   [
                       {"lat": 54.99770587584445, "lon": 82.79502868652345},
                       {"lat": 54.99928130973027, "lon": 82.92137145996095},
                       {"lat": 55.04533538802211, "lon": 82.98179626464844},
                       {"lat": 55.072470687600536, "lon": 83.04634094238281}
                   ]
    :param sources: Список индексов источников (например, [0, 1]).
    :param targets: Список индексов целей (например, [2, 3]).
    :return: Ответ API в формате JSON.
    """
    api_key = "bce701c2-be1a-4585-813c-43e20b48d2c7"
    url = f"https://routing.api.2gis.com/get_dist_matrix?key={api_key}&version=2.0"
    
    points.insert(0,base)
    sources = [0]
    targets = list(range(1, len(points)))
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "points": points,
        "type": "statistics",
        "sources": sources,
        "targets": targets,
        "start_time": start_time
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        print("Ошибка выполнения запроса:", e)
        return None
    
    try:
        return response.json()
    except json.JSONDecodeError as e:
        return None


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

def temporal_penalty(order1, order2):
    block1 = int(order1.delivery_start.hour)
    block2 = int(order2.delivery_start.hour)
    return abs(block1 - block2) * 2

def compute_combined_distance_matrix(orders, alpha, beta):
    coords = [(order.lat, order.lon) for order in orders]
    
    
    matrices = get_distance_duration_matrix(coords)
    
    spatial_matrix = np.array(matrices.get('distances'), dtype=float)
    duration_matrix = np.array(matrices.get('durations'), dtype=float)
    
    n = len(orders)
    combined = np.copy(spatial_matrix)
    combined += alpha * duration_matrix
    for i in range(n):
        for j in range(n):
            combined[i][j] += beta * temporal_penalty(orders[i], orders[j])
    return combined

def cluster_orders(
    orders,
    n_clusters,
    alpha,
    beta,
    capacity_weight=625,
    capacity_volume=2.54
):
    """
    Кластеризуем, соблюдая:
      - вместимость по весу/объёму,
      - минимальный размер кластера >= 2.
    """
    # Если заказов меньше 2 — единственный кластер из одной точки
    if len(orders) < 2:
        return {
            "labels": np.zeros(len(orders), dtype=int),
            "medoid_indices": [0],
        }
    deep_while = 0
    while True:
        # 1) строим комбинированную матрицу (расстояния + duration*alpha + time_penalty*beta)
        combined_matrix = compute_combined_distance_matrix(orders, alpha, beta)

        # 2) запускаем KMedoids
        kmedoids = KMedoids(
            n_clusters=n_clusters,
            metric='precomputed',
            random_state=42
        )
        labels = kmedoids.fit_predict(combined_matrix)

        # 3) проверяем вместимость кластеров
        capacity_ok = True
        for cl in set(labels):
            cluster_list = [o for o, lbl in zip(orders, labels) if lbl == cl]
            w = sum(float(o.weight) for o in cluster_list)
            v = sum(float(o.volume) for o in cluster_list)
            if w > capacity_weight or v > capacity_volume:
                capacity_ok = False
                break

        # 4) проверяем, что нет «одиночек»
        size_ok = all(
            list(labels).count(cl) >= 2
            for cl in set(labels)
        )

        # 5) если всё хорошо — выходим
        if capacity_ok and size_ok:
            break

        # 6) иначе корректируем число кластеров
        #    если не вмещаются — добавляем кластер
        if not capacity_ok:
            n_clusters += 1
            deep_while += 1
        #    если есть одиночки — убираем кластер
        elif not size_ok and n_clusters > 1:
            n_clusters -= 1
            deep_while += 1
        else:
            # больше ничего не изменить — выходим с тем, что есть
            break

    print("labels, kmedoids",labels, kmedoids.medoid_indices_.tolist())
    return {
        "labels": labels,
        "medoid_indices": kmedoids.medoid_indices_.tolist(),
    }

# def cluster_orders(orders, n_clusters, alpha, beta, capacity_weight=100, capacity_volume=50):
#     while True:
#         combined_matrix = compute_combined_distance_matrix(orders, alpha, beta)
#         kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42) 
#         labels = kmedoids.fit_predict(combined_matrix)
#         valid = True
#         unique_labels = set(labels)
#         for cluster in unique_labels:
#             cluster_orders_list = [order for order, label in zip(orders, labels) if label == cluster]
#             total_weight = sum(float(order.weight) for order in cluster_orders_list)
#             total_volume = sum(float(order.volume) for order in cluster_orders_list)
#             if total_weight > capacity_weight or total_volume > capacity_volume:
#                 valid = False
#                 print(f"Кластер {cluster} перегружен: вес {total_weight}, объем {total_volume}")
#         if valid:
#             break
#         else:
#             n_clusters += 1
#             print("Увеличиваем число кластеров до", n_clusters)
#     return {"labels": labels, "medoid_indices": kmedoids.medoid_indices_}

# def time_to_minutes(t: time) -> int:
#     return t.hour * 60 + t.minute





# def solve_vrptw_distance(orders, vehicle_capacity_weight, vehicle_capacity_volume, time_limit_s=30):
#     """
#     orders — уже в порядке, в котором мы хотим их попробовать;
#              функция не будет их больше сортировать.
#     """
#     # 1) Собираем матрицы, окна, деманды
#     coords    = [(o.lon, o.lat) for o in orders]
#     durations, distances = compute_osrm_matrices(coords)
    
#     t0 = to_min(orders[0].delivery_start)
#     time_windows = [(max(to_min(o.delivery_start)-t0,0),
#                      max(to_min(o.delivery_end)  -t0,0))
#                     for o in orders]
#     demands = [o.weight for o in orders]
#     volumes = [o.volume for o in orders]
#     N, depot, max_time = len(orders), 0, 12*60

#     # # 2) OSRM-тур как initial hint
#     osrm_order = get_osrm_trip_order(coords)
#     # if osrm_order[0] != 0:  # убедимся, что депо первый
#     #     osrm_order.insert(0, 0)
#     expected_next = {u: v for u,v in zip(osrm_order, osrm_order[1:])}

#     # 3) Модель
#     mgr     = pywrapcp.RoutingIndexManager(N, 1, depot)
#     routing = pywrapcp.RoutingModel(mgr)

#     # после получения osrm_order и матрицы distances:
#     # 1) считаем длины каждого OSRM-дуги
#     arc_dists = [
#         distances[u][v]
#         for u, v in zip(osrm_order, osrm_order[1:])
#     ]
#     # 2) средняя дуга (в метрах)
#     avg_arc = sum(arc_dists) / len(arc_dists)
#     # 3) назначаем penalty = avg_arc
#     penalty = avg_arc

#     # pure distance cost
#     def dist_cb(i, j):
#         u = mgr.IndexToNode(i)
#         v = mgr.IndexToNode(j)
#         base = int(distances[u][v])
#         # если не по OSRM → прибавляем штраф avg_arc
#         if expected_next.get(u) != v:
#             base += int(penalty)
#         return base
#     dist_idx = routing.RegisterTransitCallback(dist_cb)
#     routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

#     # TimeDimension с ожиданием
#     def time_cb(i,j):
#         return int(durations[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
#     time_idx = routing.RegisterTransitCallback(time_cb)
#     routing.AddDimension(time_idx, max_time, max_time, True, "Time")
#     tdim = routing.GetDimensionOrDie("Time")
#     for node in range(N):
#         idx = mgr.NodeToIndex(node)
#         if node == depot:
#             tdim.CumulVar(idx).SetRange(0, max_time)
#         else:
#             start, _ = time_windows[node]
#             tdim.CumulVar(idx).SetRange(start, max_time)

#     # Capacity & Volume
#     cap_idx = routing.RegisterUnaryTransitCallback(lambda i: demands[mgr.IndexToNode(i)])
#     routing.AddDimension(cap_idx, 0, int(vehicle_capacity_weight), True, "Capacity")
#     vol_idx = routing.RegisterUnaryTransitCallback(lambda i: volumes[mgr.IndexToNode(i)])
#     routing.AddDimension(vol_idx, 0, int(vehicle_capacity_volume), True, "Volume")

#     # # 4) Warm-start hint: подскажем NextVar согласно osrm_order
#     # for a, b in zip(osrm_order, osrm_order[1:]):
#     #     from_idx = mgr.NodeToIndex(a)
#     #     to_idx   = mgr.NodeToIndex(b)
#     #     routing.AddHint( routing.NextVar(from_idx), to_idx )

#     # 5) Поисковые параметры
#     params = pywrapcp.DefaultRoutingSearchParameters()
#     params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
#     params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
#     params.time_limit.seconds = time_limit_s

#     # 6) Решаем
#     sol = routing.SolveWithParameters(params)
#     if not sol:
#         return None

#     # 7) Извлекаем маршрут
#     idx = routing.Start(0)
#     route = []
#     while not routing.IsEnd(idx):
#         node = mgr.IndexToNode(idx)
#         route.append(orders[node].name)
#         idx = sol.Value(routing.NextVar(idx))
#     return route

# def solve_vrptw_ortools(orders, vehicle_capacity_weight, vehicle_capacity_volume, time_limit_s=30):
#     """Правильный VRPTW-решатель по официальному примеру OR-Tools."""
#     # 1) Построим data_model
#     def to_min(t):
#         try:
#             return t.hour * 60 + t.minute
#         except:
#             h, m = map(int, str(t).split(":")[:2])
#             return h * 60 + m

#     coords = [(o.lon, o.lat) for o in orders]
#     durations, distances = compute_osrm_matrices(coords)
#     durations = np.ceil(durations / 60.0).astype(int)
#     # матрицы в целых секундах/метрах
#     time_matrix = (durations + 30).astype(int)  # округляем вверх для надёжности
#     dist_matrix = distances.astype(int)

#     # time windows в минутах
#     t0 = to_min(orders[0].delivery_start)
#     time_windows = []
#     for o in orders:
#         start = max(to_min(o.delivery_start) - t0, 0)
#         end   = max(to_min(o.delivery_end)   - t0, 0)
#         time_windows.append((start, end))

#     demands = [int(o.weight) for o in orders]
#     capacities = [int(vehicle_capacity_weight)]
#     N = len(orders)
#     depot = 0

#     # 2) Менеджер и модель
#     manager = pywrapcp.RoutingIndexManager(N, len(capacities), depot)
#     routing = pywrapcp.RoutingModel(manager)

#     # 3) Транспортная стоимость (расстояние)
#     def distance_callback(from_index, to_index):
#         return int(dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])
#     dist_cb_idx = routing.RegisterTransitCallback(distance_callback)
#     routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

#     # 4) TimeDimension (windows + waiting)
#     def time_callback(from_index, to_index):
#         return int(time_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])
#     time_cb_idx = routing.RegisterTransitCallback(time_callback)
#     # максимум кумулятивного времени — самый поздний конец окна + путь
#     max_tw = max(end for _, end in time_windows)
#     routing.AddDimension(
#         time_cb_idx,
#         max_tw,      # slack: сколько минут можно ждать
#         max_tw,      # cap: максимум суммарного времени
#         True,        # fix_start_cumul_to_zero: отправление из депо = t=0
#         "Time"
#     )
#     time_dimension = routing.GetDimensionOrDie("Time")
#     # Устанавливаем жёсткие окна для каждого узла
#     for idx, window in enumerate(time_windows):
#         index = manager.NodeToIndex(idx)
#         time_dimension.CumulVar(index).SetRange(window[0], window[1])

#     # 5) CapacityDimension
#     def demand_callback(from_index):
#         return demands[manager.IndexToNode(from_index)]
#     demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
#     routing.AddDimensionWithVehicleCapacity(
#         demand_idx,
#         0,                      # no slack
#         capacities,            # вместимость каждого ТС
#         True,                   # start cumul to zero
#         "Capacity"
#     )

#     # 6) Параметры поиска
#     search_params = pywrapcp.DefaultRoutingSearchParameters()
#     search_params.first_solution_strategy = \
#         routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
#     search_params.local_search_metaheuristic = \
#         routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
#     search_params.time_limit.seconds = time_limit_s
#     search_params.log_search = True

#     # 7) Решаем
#     solution = routing.SolveWithParameters(search_params)
#     if solution is None:
#         return None

#     # 8) Извлечение маршрута
#     index = routing.Start(0)
#     route = []
#     while not routing.IsEnd(index):
#         node = manager.IndexToNode(index)
#         route.append(orders[node].name)
#         index = solution.Value(routing.NextVar(index))
#     return route

# def solve_vrptw_hybrid(orders, vehicle_capacity_weight, vehicle_capacity_volume, time_limit_s=30):
#     """
#     Сначала группируем по началу окна, потом в каждой группе — OSRM trip,
#     затем склеиваем группы по возрастанию времени старта и пушим в solve_vrptw_distance.
#     """
#     # 1) Группировка по delivery_start
#     groups = {}
#     for o in orders:
#         start = to_min(o.delivery_start)
#         groups.setdefault(start, []).append(o)
#     sorted_starts = sorted(groups)
#     print("sorted_starts, groups", sorted_starts,groups)
#     # 2) В каждой группе — OSRM trip по её координатам
#     ordered = []
#     for start in sorted_starts:
#         grp = groups[start]
#         coords = [(o.lon,o.lat) for o in grp]
#         if len(coords) > 1:
#             osrm_order = get_osrm_trip_order(coords)
#         else:
#             osrm_order = [0]
#         print(osrm_order)
#         for local_idx in osrm_order:
#             ordered.append(grp[local_idx])
#             print("ordered", ordered)

#     # 3) Запускаем чистый вариант на уже упорядоченном списке
#     return solve_vrptw_ortools(
#         ordered,
#         vehicle_capacity_weight,
#         vehicle_capacity_volume,
#         time_limit_s
#     )

def create_time_windows(orders_cluster):
    windows = []
    for order in orders_cluster:
        start = time_to_minutes(order.delivery_start)
        end = time_to_minutes(order.delivery_end)
        windows.append((start, end))
    return windows

def compute_travel_time_matrix(orders_cluster):
    coords = [(order.lat, order.lon) for order in orders_cluster]
    matrices = get_distance_duration_matrix(coords)
    duration_matrix = np.array(matrices.get('durations')) / 60.0
    return duration_matrix.tolist()

def OR_Tools_TSP(coords):
    # 1) Получаем distance_matrix из OSRM table
    durations, distances = compute_osrm_matrices(coords)  # distances в метрах

    # 2) Строим точный TSP
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    N = len(coords)
    mgr = pywrapcp.RoutingIndexManager(N, 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    def dist_cb(i,j):
        return int(distances[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
    idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search.time_limit.seconds = 5
    sol = routing.SolveWithParameters(search)
    route = []
    i = routing.Start(0)
    while not routing.IsEnd(i):
        route.append(mgr.IndexToNode(i))
        i = sol.Value(routing.NextVar(i))
    return route


def solve_hybrid_final(orders, vehicle_capacity_weight, vehicle_capacity_volume, ALPHA = 1, BETA= 25000, time_limit_s=10):
    """
    Гибрид: сначала пробуем чистый OSRM-TSP (минимум по distance + проверка окон).
    Если проходит — возвращаем OSRM-маршрут. Иначе запускаем полноценный VRPTW
    с жёсткими time windows и двумя CapacityDimension (weight и volume).
    """
    import numpy as np
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    def choose_start_index(orders):
        """
        Выбирает индекс заказа, с которого выгоднее стартовать OSRM-trip:
        1) отбираем заказы с минимальным delivery_start
        2) среди них — тот, у которого максимальная сумма расстояний до всех остальных
        """
        # 1) переводим окна в минуты
        def to_min(t):
            try:
                return t.hour * 60 + t.minute
            except:
                h,m = map(int, str(t).split(":")[:2])
                return h*60 + m
    
        # 1) минимальное время старта
        starts = [to_min(o.delivery_start) for o in orders]
        min_start = min(starts)
        candidates = [i for i, s in enumerate(starts) if s == min_start]
        if len(candidates) == 1:
            return candidates[0]

        # 2) матрица расстояний через OSRM
        coords = [(o.lon, o.lat) for o in orders]
        _, dist_matrix = compute_osrm_matrices(coords)
        print("chose_start")
        dist_matrix = dist_matrix.astype(float)

        # 3) среди кандидатов выбираем максимально удалённого
        sums = {i: dist_matrix[i].sum() for i in candidates}
        best = max(sums, key=sums.get)
        # print("best", best, "orders[best]", orders[best])
        return best

    # 1) Матрицы OSRM
    start_idx = choose_start_index(orders)
    orders = [orders[start_idx]] + [o for i,o in enumerate(orders) if i != start_idx]
    coords = [(o.lon, o.lat) for o in orders]
    durations, distances = compute_osrm_matrices(coords)
    # переводим секунды → минуты, округляя вверх
    durations = np.ceil(durations / 60.0).astype(int)

    
    # 2) Чистый OSRM-тур по всем точкам
    # osrm_order = get_osrm_trip_order(coords)
    osrm_order = OR_Tools_TSP(coords)
    print("osrm_order", "\n",osrm_order,"\n",[orders[i].name for i in osrm_order])
    # # гарантируем, что депо (индекс 0) в начале
    # if osrm_order[0] != 0:
    #     osrm_order.insert(0, 0)

    # функция проверки физических окон
    def check_osrm(orders, osrm_order, durations):
        def to_min(t):
            try: return t.hour * 60 + t.minute
            except: h,m = map(int,str(t).split(":")[:2]); return h*60 + m

        # собираем окна в минутах
        wins = [(to_min(o.delivery_start), to_min(o.delivery_end)) for o in orders]
        # стартуем ровно в начале первого окна
        t = wins[osrm_order[0]][0]
        for u, v in zip(osrm_order, osrm_order[1:]):
            t += durations[u][v]
            s, e = wins[v]
            if t < s:
                t = s
            if t > e:
                return False
        return True

    # 3) Если OSRM-маршрут укладывается — возвращаем его
    if check_osrm(orders, osrm_order, durations):
        return [orders[i].name for i in osrm_order]

    # 4) Иначе — строим VRPTW
    def to_min(t):
        try: return t.hour * 60 + t.minute
        except: h,m = map(int,str(t).split(":")[:2]); return h*60 + m

    # Матрицы для VRPTW
    time_matrix = durations  # в минутах
    dist_matrix = distances.astype(int)

    # time windows относительно первого
    t0 = to_min(orders[0].delivery_start)
    time_windows = [
        (max(to_min(o.delivery_start) - t0, 0),
         max(to_min(o.delivery_end)   - t0, 0))
        for o in orders
    ]

    # demands по весу и объёму
    weight_demands = [int(o.weight) for o in orders]
    volume_demands = [int(o.volume) for o in orders]  # например, переводим м³→л или кг, как вам удобнее

    N, depot = len(orders), 0

    mgr     = pywrapcp.RoutingIndexManager(N, 1, depot)
    routing = pywrapcp.RoutingModel(mgr)

    # 4.1) Cost = расстояние
    def dist_cb(i, j):
        return int(dist_matrix[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    # 4.2) TimeDimension (windows + slack)
    def time_cb(i, j):
        return int(time_matrix[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
    time_idx = routing.RegisterTransitCallback(time_cb)
    max_tw = max(e for _, e in time_windows)
    routing.AddDimension(
        time_idx,
        max_tw,   # slack (ожидание) в минутах
        max_tw,   # cap: максимум всего времени
        True,     # fix_start_cumul_to_zero
        "Time"
    )
    td = routing.GetDimensionOrDie("Time")
    for node, (s, e) in enumerate(time_windows):
        idx = mgr.NodeToIndex(node)
        td.CumulVar(idx).SetRange(s, e)

    # 4.3) CapacityDimension по весу
    def weight_cb(i):
        return weight_demands[mgr.IndexToNode(i)]
    w_idx = routing.RegisterUnaryTransitCallback(weight_cb)
    routing.AddDimension(
        w_idx,
        0,                            # no slack
        int(vehicle_capacity_weight), # capacity
        True,                         # start cumul to zero
        "CapacityWeight"
    )

    # 4.4) CapacityDimension по объёму
    def volume_cb(i):
        return volume_demands[mgr.IndexToNode(i)]
    v_idx = routing.RegisterUnaryTransitCallback(volume_cb)
    routing.AddDimension(
        v_idx,
        0,
        int(vehicle_capacity_volume * 1000),  # переведите в ту же единицу, что и volume_demands
        True,
        "CapacityVolume"
    )

    # 5) Параметры поиска
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s
    params.log_search = True

    sol = routing.SolveWithParameters(params)
    if not sol:
        print(f"[!] Не найден маршрут для кластера из {N} заказов. Запускается пере-кластеризация...")

        # Попытка деления на 2 подкластера
        subclusters_info = cluster_orders(orders, n_clusters=2,
                                          alpha=ALPHA, beta=BETA,
                                          capacity_weight=vehicle_capacity_weight,
                                          capacity_volume=vehicle_capacity_volume)
        new_labels = subclusters_info["labels"]

        subcluster1 = [order for order, label in zip(orders, new_labels) if label == 0]
        subcluster2 = [order for order, label in zip(orders, new_labels) if label == 1]

        if len(subcluster1) < 2 or len(subcluster2) < 2:
            print("[!] Один из подкластеров слишком мал для маршрутизации. Отмена.")
            return None

        route1 = solve_hybrid_final(subcluster1, vehicle_capacity_weight, vehicle_capacity_volume, ALPHA, BETA)
        route2 = solve_hybrid_final(subcluster2, vehicle_capacity_weight, vehicle_capacity_volume, ALPHA, BETA)

        if route1 is None or route2 is None:
            print("[!] Не удалось составить маршруты даже после пере-кластеризации.")
            return None

        # Возврат как объединённый результат
        return route1 + route2 if isinstance(route1, list) and isinstance(route2, list) else [*route1, *route2]

    # 6) Извлечение маршрута
    idx   = routing.Start(0)
    route = []
    while not routing.IsEnd(idx):
        node = mgr.IndexToNode(idx)
        route.append(orders[node].name)
        idx = sol.Value(routing.NextVar(idx))
    return route

def solve_vrptw(orders_cluster, orders_df, vehicle_capacity_weight, vehicle_capacity_volume, ALPHA, BETA):

    def get_demands(orders_cluster):
        return [order.weight for order in orders_cluster]

    def get_volumes(orders_cluster):
        return [order.volume for order in orders_cluster]

    def nearest_neighbor_order(orders, time_matrix, start_index=0):
        """
        Простая эвристика: начинаем с точки start_index (депо или самой ранней точки),
        а дальше каждый раз выбираем ещё не посещённую точку с минимальным временем в пути.
        
        :param orders:    список заказов (или любых объектов), len = N
        :param time_matrix:  N×N numpy-массив с временами в пути между ними
        :param start_index:  индекс начальной точки в списке orders (обычно 0)
        :return: список индексов 0..N-1 в порядке обхода
        """
        N = len(orders)
        unvisited = set(range(N))
        sequence = []
        current = start_index
        while unvisited:
            sequence.append(current)
            unvisited.remove(current)
            if not unvisited:
                break
            # выбираем ближайшую среди ещё не посещённых
            next_idx = min(unvisited, key=lambda j: time_matrix[current][j])
            current = next_idx
        return sequence


    # # Сортируем заказы по delivery_start
    # orders_cluster_sorted = sorted(orders_cluster, key=lambda order: order.delivery_start)

    # 1) отсортируем по времени начала, чтобы выбрать depot (точку с наименьшим start)
    orders_cluster_sorted = sorted(orders_cluster, key=lambda o: o.delivery_start)
    num_orders = len(orders_cluster_sorted)

    # 2) посчитаем матрицу времени в пути
    time_matrix = compute_travel_time_matrix(orders_cluster_sorted)

    # 3) получим эвристический порядок индексов
    init_seq = nearest_neighbor_order(orders_cluster_sorted, time_matrix, start_index=0)

    # 4) переставляем сами заказы в этом порядке
    orders_cluster_sorted = [orders_cluster_sorted[i] for i in init_seq]

    num_orders = len(orders_cluster_sorted)
    depot = 0
    
    
    t0 = time_to_minutes(orders_cluster_sorted[0].delivery_start)
    time_windows = []
    for order in orders_cluster_sorted:
        start = max(time_to_minutes(order.delivery_start) - t0, 0)
        end = max(time_to_minutes(order.delivery_end) - t0, 0)
        time_windows.append((start, end))
    
    # time_matrix = compute_travel_time_matrix(orders_cluster_sorted)
    demands = get_demands(orders_cluster_sorted)
    volumes = get_volumes(orders_cluster_sorted)
    max_time = 12 * 60
    
    manager = pywrapcp.RoutingIndexManager(num_orders, 1, depot)
    routing = pywrapcp.RoutingModel(manager)
    
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix[from_node][to_node])
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    time_dimension_name = "Time"
    routing.AddDimension(
        transit_callback_index,
        30,
        max_time,
        False,
        time_dimension_name
    )
    time_dimension = routing.GetDimensionOrDie(time_dimension_name)
    
    for node in range(num_orders):
        index = manager.NodeToIndex(node)
        if node == depot:
            time_dimension.CumulVar(index).SetRange(0, max_time)
        else:
            start, end = time_windows[node]
            time_dimension.CumulVar(index).SetRange(start, end)
    
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(demands[from_node])
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimension(
        demand_callback_index,
        0,
        int(vehicle_capacity_weight),
        True,
        "Capacity"
    )
    
    def volume_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(volumes[from_node])
    volume_callback_index = routing.RegisterUnaryTransitCallback(volume_callback)
    routing.AddDimension(
        volume_callback_index,
        0,
        int(vehicle_capacity_volume),
        True,
        "Volume"
    )
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION
    # search_parameters.time_limit.seconds = 30
    # search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
 
    # search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    # search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30
    
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        print("Не найдено решение для VRPTW для кластера с", num_orders, "заказами, пере-кластеризую заказы")
        # Пере-кластеризуем набор заказов внутри данного кластера:
        subclusters_info = cluster_orders(orders_cluster_sorted, n_clusters=2,
                                          alpha=ALPHA, beta=BETA,
                                          capacity_weight=vehicle_capacity_weight,
                                          capacity_volume=vehicle_capacity_volume)
        new_labels = subclusters_info["labels"]
        # Определяем глобальное максимальное значение кластеров в orders_df и назначаем новые
        global_max = orders_df['cluster'].max() if not orders_df['cluster'].empty else 0
        for order, label in zip(orders_cluster_sorted, new_labels):
            new_global_label = orders_cluster[0]['cluster'] if label == 0 else global_max + 1
            orders_df.at[order.name, 'cluster'] = new_global_label
        subcluster1 = [order for order, label in zip(orders_cluster_sorted, new_labels) if label == 0]
        subcluster2 = [order for order, label in zip(orders_cluster_sorted, new_labels) if label == 1]
        
        route1 = solve_vrptw(subcluster1, orders_df, vehicle_capacity_weight, vehicle_capacity_volume,ALPHA, BETA)
        route2 = solve_vrptw(subcluster2, orders_df, vehicle_capacity_weight, vehicle_capacity_volume, ALPHA, BETA)
        if route1 is None or route2 is None:
            print("Не найдено решение даже после пере-кластеризации")
            return None
        if isinstance(route1, tuple):
            ret_route1 = route1
        else: 
            ret_route1 = {orders_cluster[0]['cluster'] : route1}

        if isinstance(route2, tuple):
            ret_route2 = route2
        else: 
            ret_route2 = {global_max + 1 : route2} 

        return ret_route1, ret_route2
    
    # Извлекаем маршрут и возвращаем номера заказов (order_number)
    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        elem = orders_cluster_sorted[node]
        # Определяем идентификатор заказа: если это модель Order, берем .order_number, иначе для pandas.Series — name
        if hasattr(elem, 'order_number'):
            order_number = elem.order_number
        else:
            order_number = elem.name
        route.append(order_number)
        index = solution.Value(routing.NextVar(index))
    return route

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

   
def flatten_route_element(x):
    if isinstance(x, int):
        return [x]
    elif isinstance(x, (list, tuple)):
        result = []
        for item in x:
            result.extend(flatten_route_element(item))
        return result
    else:
        return []
    
# def add_route_with_labels(route_poly_coords, cluster, cluster_to_color, map_obj, route_coords):
#     """
#     Добавляет на карту маршрут с указанием порядковых номеров точек и стрелками, показывающими направление.
#     Аргументы:
#       route_coords (list): Список координат маршрута, где каждая координата имеет вид (lat, lon).
#       cluster (int или str): Номер кластера, для подписи.
#       cluster_to_color (dict): Словарь с цветами для каждого кластера.
#       map_obj: Объект folium.Map.
#     """
#     # Добавляем полилинию маршрута
#     polyline = folium.PolyLine(route_poly_coords, color=cluster_to_color.get(cluster, 'gray'), weight=3)
#     polyline.add_to(map_obj)
    
#     # Добавляем стрелки вдоль маршрута для указания направления
#     PolyLineTextPath(
#         polyline, 
#         '   ▶   ',         # текст для отображения (символ стрелки)
#         repeat=True,        # повторяем стрелку по всей длине
#         offset=7,           # смещение стрелки от линии
#         attributes={'font-size': '14', 'fill': cluster_to_color.get(cluster, 'gray')}
#      ).add_to(map_obj)
#
#     # Добавляем маркировку точек (с номерами) вдоль маршрута
#     for i, coord in enumerate(route_coords):
#         folium.Marker(
#             location=coord,
#             icon=folium.DivIcon(
#                 html=f'<div style="font-size: 12pt; color: black; pointer-events: none;"">{i+1}</div>',
#                 icon_anchor=(0, -8)
#             )
#         ).add_to(map_obj)
    
def add_route(route,  cluster, cluster_to_color, map_obj, cluster_df):
    
    route_coords = []
    for idx in route:
        if isinstance(idx, int) and idx <= max(cluster_df['order_number']):
            row = cluster_df.loc[cluster_df['order_number'] == idx]
            route_coords.append((float(row['lat']), float(row['lon'])))
    
    route_poly_osrm = get_route(route_coords)
    
    if route_poly_osrm:
        add_route_with_labels(route_poly_osrm.get('geometry_pol'), cluster, cluster_to_color, map_obj,route_coords)


def perform_st_clustering(orders, eps_space=0.5, eps_time=1.0, min_samples=2):
    """
    Выполняет спатиально-временную кластеризацию заказов (ST-DBSCAN).
    
    :param orders: Список заказов. Для каждого заказа предполагается наличие:
                   - order.address: строка с адресом,
                   - order.delivery_start: объект времени (datetime.time).
    :param eps_space: Порог для пространственного расстояния (например, в градусах).
    :param eps_time:  Порог для временного расстояния (например, в часах).
    :param min_samples: Минимальное количество соседей для формирования кластера.
    :return: Массив кластерных меток, соответствующий порядку заказов.
    """
    # Функция для перевода времени в числовое значение (часы в виде float)
    def time_to_float(t):
        return t.hour + t.minute / 60.0
    
    # Формируем набор данных, где каждая точка представлена как [lat, lon, time]
    data = []
    for order in orders:
        # Получаем координаты через функцию geocode
        lat, lon = geocode(order.address)
        # Преобразуем время доставки (delivery_start) в число (час)
        t_val = time_to_float(order.delivery_start)
        data.append([lat, lon, t_val])
    data = np.array(data)
    
    # Определяем собственную метрику, которая объединяет пространственную и временную дистанции.
    def st_metric(u, v):
        # Пространственное расстояние (для небольших площадей можно использовать Евклидово расстояние)
        spatial_distance = np.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)
        # Временное расстояние (в часах)
        temporal_distance = abs(u[2] - v[2])
        # Если хотя бы один компонент превышает порог, точки не считаются соседями:
        if spatial_distance > eps_space or temporal_distance > eps_time:
            # Возвращаем значение больше суммы порогов, чтобы гарантировать, 
            # что сравнение с общим eps (будем использовать eps=(eps_space + eps_time)) не пройдёт.
            return eps_space + eps_time + 1
        # Иначе возвращаем суммарное расстояние как "композитную" меру.
        return spatial_distance + temporal_distance
    
    # Задаем общее значение порога для DBSCAN как сумму индивидуальных порогов.
    composite_eps = eps_space + eps_time
    
    db = DBSCAN(eps=composite_eps, min_samples=min_samples, metric=st_metric)
    clusters = db.fit_predict(data)
    return clusters

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

def temporal_penalty(order1, order2):
    """
    Вычисляет штраф за временную разницу между заказами.
    
    Предполагается, что order.delivery_start — объект datetime.time.
    Преобразуем время в номер двухчасового блока:
        block = int(delivery_start.hour // 2)
    Если заказы в одном блоке → штраф = 0,
    иначе штраф равен абсолютной разности блоков, умноженной на величину блока (2 часа).
    """
    # Преобразуем время доставки в номер блока (например, 12:00-14:00 → block = 6 // 2 = 3)
    block1 = int(order1.delivery_start.hour // 2)
    block2 = int(order2.delivery_start.hour // 2)
    return abs(block1 - block2) * 2   # штраф в часах; можно масштабировать

def compute_combined_distance_matrix(orders, alpha=1.0, beta=1000.0):
    """
    Вычисляет комбинированную матрицу расстояний с учетом:
     - Пространственной составляющей (расстояния в метрах),
     - Матрицы времени поездки (в секундах),
     - Штрафа за разницу двухчасовых интервалов (в часах, затем умножается на β).
    
    Аргументы:
      orders (list): Список заказов, у которых предварительно заполнены координаты 
                     (order.latitude, order.longitude) и время доставки (order.delivery_start).
      alpha (float): Весовой коэффициент для времени поездки (матрица durations).
      beta (float): Весовой коэффициент для временного штрафа (единицы приводятся в метрический масштаб).
                  Например, если β = 1000, разница в 1 час (между разными блоками) прибавит 1000 метров.
    
    Возвращает:
      combined (np.array): N x N матрица объединенного расстояния.
    """
    # Собираем координаты заказов в виде списка кортежей (lat, lon)
    coords = [(order.lat, order.lon) for order in orders]
    matrices = get_distance_duration_matrix(coords)
    
    # Получаем пространственную матрицу и матрицу времени поездки
    spatial_matrix = np.array(matrices.get('distances'))  # в метрах
    duration_matrix = np.array(matrices.get('durations'))  # в секундах
    
    n = len(orders)
    combined = np.copy(spatial_matrix)  # начнем с пространственной составляющей
    
    # Преобразуем время поездки: например, если оставить в секундах,
    # коэффициент α можно подобрать так, чтобы α*duration_matrix имел размеры, сопоставимые с метрами.
    # Здесь предполагаем, что α может быть подобран эмпирически.
    combined += alpha * duration_matrix  # прибавляем вклад времени поездки
    
    # Добавляем штраф по времени доставки, основанный на различии двухчасовых интервалов
    for i in range(n):
        for j in range(n):
            time_penalty = temporal_penalty(orders[i], orders[j])
            combined[i][j] += beta * time_penalty
    return combined



# def cluster_orders(orders, n_clusters, alpha=1.0, beta=1000.0):
    """
    Кластеризует заказы, учитывая:
      1. Матрицу расстояний между точками (полученную от OSRM),
      2. Время поездки между точками,
      3. Временной штраф за несоответствие двухчасового интервала доставки,
      4. Число кластеров, равное количеству курьеров.
    
    Аргументы:
      orders (list): Список заказов, для которых предварительно заполнены координаты 
                     (latitude, longitude) и время доставки (delivery_start).
      n_clusters (int): Количество кластеров (обычно равно количеству курьеров).
      alpha (float): Коэффициент для матрицы времени (duration_matrix).
      beta (float): Коэффициент для временного штрафа.
    
    Возвращает:
      labels (np.ndarray): Массив меток кластеров (индекс для каждого заказа).
    """
    combined_matrix = compute_combined_distance_matrix(orders, alpha, beta)
    
    # Используем K-medoids с метрикой 'precomputed'
    print("n_clusters",n_clusters)
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(combined_matrix)
    return {"labels" : labels, "medoid_indices" : kmedoids.medoid_indices_}
