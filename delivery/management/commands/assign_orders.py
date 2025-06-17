import logging
import pandas as pd
from django.core.management.base import BaseCommand
from django.db import transaction
from delivery.models import Order, Courier, RouteItem, Route
from delivery.services.utils import * # geocluster, assign_couriers_to_clusters, orders_to_dataframe
from delivery.services.clustering_solveVRPTW import solve_vrptw, geocluster, solve_hybrid_final # solve_vrptw_distance, solve_vrptw_hybrid, 

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Первичная кластеризация, оптимизация маршрутов и назначение курьеров'

    def add_arguments(self, parser):
        parser.add_argument(
            '--vehicle_capacity_weight',
            type=float,
            help='Грузоподъёмность для маршрутизации (кг)'
        )
        parser.add_argument(
            '--vehicle_capacity_volume',
            type=float,
            help='Объём багажника для маршрутизации (м³)'
        )
        parser.add_argument(
            '--alpha',
            type=float,
            help='Коэффициент α для кластеризации'
        )
        parser.add_argument(
            '--beta',
            type=float,
            help='Коэффициент β для кластеризации'
        )

    def handle(self, *args, **options):
        # Параметры маршрутизации из опций (или стандартные)
        vcw   = options.get('vehicle_capacity_weight')
        vcv   = options.get('vehicle_capacity_volume')
        alpha = options.get('alpha') or 0
        beta  = options.get('beta')  or 10000
        
        # 1. Получаем доступных курьеров
        available = list(Courier.objects.filter(is_available=True))
        n_couriers = len(available)
        if n_couriers == 0:
            self.stdout.write(self.style.WARNING('Нет доступных курьеров.'))
            return

        # 2. Отбираем заказы 
        orders_qs = Order.objects.all()
        orders_with_cluster = Order.objects.filter(cluster__isnull=True)
        if not orders_qs.exists():
            self.stdout.write(self.style.SUCCESS('Нет новых заказов для распределения.'))
            return
        # Подготовка DataFrame с актуальными кластерами
        orders_df = orders_to_dataframe()
        clusters_map = {}
        if orders_with_cluster.exists():

            # 3. Подготовка данных для кластеризации
            data = [(
                o.order_number,
                float(o.lat), float(o.lon),
                o.delivery_start.hour * 60, o.delivery_end.hour * 60,
                float(o.weight), float(o.volume)
            ) for o in orders_qs]

            # 4. Кластеризация (число кластеров = число курьеров)
            clusters_map = geocluster(data, n_clusters=n_couriers, ALPHA=alpha,BETA=beta, vcw=vcw, vcv=vcv)
            print("info", vcv, vcw, alpha, beta, clusters_map)
            self.stdout.write(self.style.SUCCESS(
                f'Кластеризация: {len(set(clusters_map.values()))} кластеров.'
            ))
                  
            # Обновляем кластеры в DataFrame из clusters_map
            orders_df['cluster'] = orders_df.index.to_series().map(lambda oid: clusters_map.get(oid))
        else:    
            for o in orders_qs:
                clusters_map.update({o.order_number:o.cluster})

        print("clusters_map_assign",  clusters_map)
        # 5. Оптимизация маршрутов внутри кластеров
        cluster_routes = {} 
        orders_in_clusters: dict[int, list] = {}
        for oid, cid in clusters_map.items():
            orders_in_clusters.setdefault(cid, []).append(oid)

        for cid, oids in orders_in_clusters.items():
            # Составляем список строк Series для solve_vrptw
            cluster_df = orders_df.loc[oids]
            orders_cluster = [row for _, row in cluster_df.iterrows()]
            # print("orders_cluster", orders_cluster)
            # Оптимизация VRPTW
            route = solve_hybrid_final(
                orders_cluster,
                # orders_df,
                vehicle_capacity_weight=vcw,
                vehicle_capacity_volume=vcv, 
                # ALPHA=alpha, BETA=beta
            )
            # try:
            #     route1 = solve_vrptw_distance(
            #         orders_cluster,
            #         # orders_df,
            #         vehicle_capacity_weight=vcw,
            #         vehicle_capacity_volume=vcv, 
            #         # ALPHA=alpha, BETA=beta
            #     )
            #     print("route1",route1)
            # except:
            #     print("route1 - ошибка", Exception())
            # try:
            #     route2 = solve_vrptw_hybrid(
            #         orders_cluster,
            #         # orders_df,
            #         vehicle_capacity_weight=vcw,
            #         vehicle_capacity_volume=vcv, 
            #         # ALPHA=alpha, BETA=beta
            #     )
            #     print("route2",route2)
            # except:
            #     print("route2 - ошибка", Exception())

            cluster_routes[cid] = route
            self.stdout.write(
                self.style.SUCCESS(f'Кластер {cid}: маршрут {route}')
            )
        print("cluster_routes", cluster_routes)
        # Обновление clusters_map на основе актуальных кластеров в DataFrame после оптимизации
        # Приводим Series [cluster] к dict: {order_number: cluster}
        clusters_map = orders_df['cluster'].to_dict()
        # 6. Назначение курьеров после оптимизации
        cluster_to_courier = assign_couriers_to_clusters(clusters_map)
        self.stdout.write(self.style.SUCCESS(
            f'Назначено курьеров: {len(cluster_to_courier)} из {n_couriers} доступных.'
        ))
        n_clusters = len(orders_df['cluster'].unique())
        if n_couriers < n_clusters:
            self.stdout.write(self.style.NOTICE(
            f'{n_clusters - n_couriers} осталось не распределенных кластеров.'
            ))
        print(clusters_map, orders_df)
        # 7. Сохранение результатов в БД
        with transaction.atomic():
            for oid, cid in clusters_map.items():
                courier = cluster_to_courier.get(cid)
                Order.objects.filter(order_number=oid).update(
                    cluster=cid,
                    courier=courier
                )
            Route.objects.all().delete()
            for cid, seq in cluster_routes.items():
                courier = cluster_to_courier.get(cid)
                if not courier:
                    continue
                
                # # Удаляем старый маршрут 
                # Route.objects.filter(
                #     courier=courier,
                #     cluster=cid,
                # ).delete()

                # Новый маршрут
                route_obj = Route.objects.create(
                    courier=courier,
                    cluster=cid,

                )

                # Позиции маршрута в порядке sequence
                for idx, order_number in enumerate(seq, start=1):
                    order = Order.objects.get(order_number=order_number)
                    RouteItem.objects.create(
                        route=route_obj,
                        order=order,
                        sequence=idx
                    )   
        self.stdout.write(self.style.SUCCESS('Сохранены кластеры и курьеры для всех заказов.'))