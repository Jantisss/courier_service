from django.urls import path
from .views import import_orders
from django.shortcuts import render
from .views import (
    import_orders, 
    LogisticsDashboardView, OrderUpdateView, 
    distribute_orders, clear_orders,
    CourierListView, CourierCreateView,
    CourierUpdateView, cluster_only, route_sheets, route_sheet_for_courier, CourierDeleteView
)


app_name = 'delivery'
urlpatterns = [
    path('import/', import_orders, name='import_orders'),
    path('import/success/', lambda r: render(r, 'delivery/import_success.html'), name='import_success'),
    # path('clusters/', clustered_map_VRPTM_ORTools, name='clusters_map'),
    # path('map/', map_VRPTM, name='map_vrptw'),
    path('logistics/', LogisticsDashboardView.as_view(), name='logistics_dashboard'),
    path('logistics/order/<int:pk>/edit/', OrderUpdateView.as_view(), name='order_edit'),
    path('logistics/distribute/', distribute_orders, name='distribute_orders'),
    path('logistics/clear/', clear_orders, name='clear_orders'),

    path('couriers/', CourierListView.as_view(),   name='courier_list'),
    path('couriers/add/', CourierCreateView.as_view(), name='courier_add'),
    path('couriers/<int:pk>/edit/', CourierUpdateView.as_view(), name='courier_edit'),
    path('couriers/<int:pk>/delete/', CourierDeleteView.as_view(), name='courier_delete'),

    path('cluster_only/', cluster_only, name='cluster_only'),
    path('route_sheets/', route_sheets, name='route_sheets'),
    path('route_sheet/<int:courier_id>/', route_sheet_for_courier, name='route_sheet_for_courier'),
]