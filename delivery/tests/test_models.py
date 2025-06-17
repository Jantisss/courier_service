# delivery/tests/test_models.py
import pytest
from django.contrib.auth.models import User
from django.db import IntegrityError, transaction
from django.utils import timezone

from delivery.models import Courier, Order, Route, RouteItem


@pytest.mark.django_db
def test_courier_str_and_one_to_one():
    # Создаём пользователя
    user = User.objects.create_user(username='ivan', password='pass')
    # Создаём профиль курьера
    courier = Courier.objects.create(
        user=user,
        login=user.username,
        fio='Иван Иванов',
        contact='+71234567890',
        vehicle_capacity_weight=100.0,
        vehicle_capacity_volume=1.5
    )
    # __str__ возвращает ФИО
    assert str(courier) == 'Иван Иванов'

    # Попытка создать второго Courier для того же User -> IntegrityError
    with pytest.raises(IntegrityError):
        Courier.objects.create(
            user=user,
            login=user.username,
            fio='Другой Курьер',
            contact='+70001112233',
            vehicle_capacity_weight=50.0,
            vehicle_capacity_volume=0.5
        )


@pytest.mark.django_db
def test_order_str():
    # Создаём минимально необходимый заказ
    o = Order.objects.create(
        address='ул. Ленина, д. 1',
        delivery_start='09:00',
        delivery_end='18:00',
        weight=5.0,
        volume=0.1,
        contact='+70009998877'
    )
    # __str__ включает номер и адрес
    assert str(o) == f'Заказ #{o.order_number}: ул. Ленина, д. 1'


@pytest.mark.django_db
def test_route_str_default_date_and_unique_together():
    user = User.objects.create_user(username='petr', password='pass')
    courier = Courier.objects.create(
        user=user,
        login=user.username,
        fio='Пётр Петров',
        contact='+70002223344',
        vehicle_capacity_weight=80.0,
        vehicle_capacity_volume=1.0
    )
    # Создаём маршрут
    r = Route.objects.create(courier=courier, cluster=1)
    # По умолчанию date == сегодняшняя дата
    assert r.date == timezone.localdate()
    # __str__ формируется как "Маршрут {cluster} – {fio}"
    assert str(r) == 'Маршрут 1 – Пётр Петров'

    # Повторное создание с тем же courier+cluster -> IntegrityError
    with pytest.raises(IntegrityError):
        Route.objects.create(courier=courier, cluster=1)


@pytest.mark.django_db
def test_routeitem_str_and_unique_together_and_ordering():
    # Подготовка: courier, route и два заказа
    user = User.objects.create_user(username='vasya', password='pass')
    courier = Courier.objects.create(
        user=user,
        login=user.username,
        fio='Василий Васильев',
        contact='+70005556677',
        vehicle_capacity_weight=70.0,
        vehicle_capacity_volume=0.8
    )
    route = Route.objects.create(courier=courier, cluster=2)
    o1 = Order.objects.create(
        address='ул. Пушкина, д. 10',
        delivery_start='10:00',
        delivery_end='12:00',
        weight=2.0,
        volume=0.05,
        contact='+70001110000'
    )
    o2 = Order.objects.create(
        address='пр. Мира, д. 5',
        delivery_start='12:00',
        delivery_end='14:00',
        weight=3.0,
        volume=0.07,
        contact='+70002220000'
    )

    # Создаём позиции маршрута
    item1 = RouteItem.objects.create(route=route, order=o1, sequence=1)
    item2 = RouteItem.objects.create(route=route, order=o2, sequence=2)

    # __str__ содержит sequence и адрес заказа
    assert str(item1) == '1. ул. Пушкина, д. 10'
    assert str(item2) == '2. пр. Мира, д. 5'

    # При попытке добавить ту же пару route+order — IntegrityError
    with pytest.raises(IntegrityError):
        with transaction.atomic():
            RouteItem.objects.create(route=route, order=o1, sequence=3)

    # Теперь транзакция откатилась, и мы можем дальше читать из БД без ошибок
    seqs = [it.sequence for it in route.items.all()]
    assert seqs == [1, 2]
