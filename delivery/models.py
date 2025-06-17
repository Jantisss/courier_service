from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class Courier(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='courier_profile',
    )
    # дублируем логин (User.username)
    login = models.CharField("Логин для авторизации", max_length=150, editable=False)
    fio = models.CharField("ФИО курьера", max_length=255)
    contact = models.CharField("Контактный телефон", max_length=20)
    vehicle_capacity_weight = models.DecimalField(
        "Грузоподъёмность (кг)", max_digits=6, decimal_places=2
    )
    vehicle_capacity_volume = models.DecimalField(
        "Объём багажника (м³)", max_digits=6, decimal_places=3
    )
    is_available = models.BooleanField(
        "Доступен для доставки",
        default=True,
        help_text="Если неактивен, заказы не будут назначаться этому курьеру"
    )

    def __str__(self):
        return self.fio
    
class Order(models.Model):
    order_number = models.AutoField(primary_key=True)
    address = models.CharField(max_length=255)
    lat = models.DecimalField(max_digits=10, decimal_places=6, null=True)
    lon = models.DecimalField(max_digits=10, decimal_places=6, null=True)
    delivery_start = models.TimeField()
    delivery_end = models.TimeField()
    weight = models.DecimalField(max_digits=6, decimal_places=2)
    volume = models.DecimalField(max_digits=6, decimal_places=3)
    contact = models.CharField(max_length=20)
    comment = models.TextField(blank=True)
    cluster = models.IntegerField(null=True)
    courier = models.ForeignKey(
        Courier,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='orders',
        verbose_name='Назначенный курьер'
    )

    def __str__(self):
        return f'Заказ #{self.order_number}: {self.address}'
    
class Route(models.Model):
    """
    Маршрут для одного курьера и одного кластера заказов на конкретную дату.
    Порядок заказов хранится в промежуточной модели RouteItem.
    """
    courier = models.ForeignKey(
        'Courier',
        on_delete=models.PROTECT,
        related_name='routes',
        verbose_name='Курьер'
    )
    cluster = models.IntegerField(
        verbose_name='Номер кластера'
    )

    date = models.DateField(
        'Дата маршрута',
        default=timezone.localdate
    )

    class Meta:
        unique_together = ('courier', 'cluster')
        ordering = ['courier', 'cluster']
        verbose_name = 'Маршрут'
        verbose_name_plural = 'Маршруты'

    def __str__(self):
        return f"Маршрут {self.cluster} – {self.courier.fio}"


class RouteItem(models.Model):
    """
    Позиция в маршруте: конкретный заказ и его порядок следования.
    """
    route = models.ForeignKey(
        Route,
        on_delete=models.CASCADE,
        related_name='items',
        verbose_name='Маршрут'
    )
    order = models.ForeignKey(
        'Order',
        on_delete=models.PROTECT,
        related_name='route_items',
        verbose_name='Заказ'
    )
    sequence = models.PositiveIntegerField(
        verbose_name='Порядок следования'
    )

    STATUS_CHOICES = [
        ('delivered',     'Доставлен'),
        ('not_delivered', 'Не доставлен'),
        ('late',          'Опоздание'),
    ]
    status = models.CharField(
        'Статус доставки',
        max_length=20,
        choices=STATUS_CHOICES,
        blank=True
    )
    courier_comment = models.TextField(
        'Комментарий курьера',
        blank=True
    )
    delivered_at = models.DateTimeField(
        'Время фактической доставки',
        null=True, blank=True
    )
    class Meta:
        unique_together = ('route', 'order')
        ordering = ['sequence']
        verbose_name = 'Позиция маршрута'
        verbose_name_plural = 'Позиции маршрута'

    def __str__(self):
        return f"{self.sequence}. {self.order.address}"