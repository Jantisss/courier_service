from django.contrib import admin
from .models import Order, Courier, Route, RouteItem

@admin.register(Courier)
class CourierAdmin(admin.ModelAdmin):
    list_display = (
        'fio', 'contact', 'vehicle_capacity_weight',
        'vehicle_capacity_volume', 'is_available'
    )
    list_filter = ('is_available',)
    search_fields = ('fio', 'contact')

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = (
        'order_number', 'address', 'delivery_start',
        'delivery_end', 'weight', 'volume', 'contact',
        'cluster', 'courier'
    )
    list_filter = ('delivery_start', 'delivery_end', 'cluster', 'courier')
    search_fields = ('address', 'contact', 'comment')
    readonly_fields = ('lat', 'lon')


