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
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION
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
