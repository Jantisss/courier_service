"""
URL configuration for courier_service project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from delivery import views as delivery_views
from django.contrib.auth import views as auth_views
from django.conf import settings
from delivery.views import CourierLoginView
from django.views.generic import RedirectView


router = routers.DefaultRouter()
router.register(r'orders', delivery_views.OrderViewSet)  # регистрируем endpoint "orders"

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('dashboard/', delivery_views.dashboard, name='dashboard'),
    # path('courier/', delivery_views.courier_dashboard, name='courier_dashboard'),
    # path('map/', delivery_views.map_view, name='map_view'),
    # path('cluster-map/', delivery_views.cluster_map, name='cluster_map'),
    path('', RedirectView.as_view(
            url='delivery/logistics',    
            permanent=False       
        ), name='home'),  # Главная страница
    path('api/', include(router.urls)),
    # path('clustered-map/', delivery_views.clustered_map_view, name='clustered_map'),
    # path('clustered-orders/', delivery_views.clustered_orders_view, name='clustered_orders'),
    # path('st-cluster-map/', delivery_views.st_cluster_dashboard, name='st-cluster-map'),
    # path('api/auto_distance_matrix/', delivery_views.auto_distance_matrix, name='auto_distance_matrix'),
    # path('clustered_map_VRPTM_ORTools/', delivery_views.clustered_map_VRPTM_ORTools, name='clustered_map_VRPTM_ORTools'),
    # path('clustered_ONLYmap_VRPTM_ORTools/', delivery_views.map_VRPTM, name='clustered_ONLYmap_VRPTM_ORTools'),
    path('delivery/', include('delivery.urls'), name="delivery"),
    path('accounts/login/', 
         CourierLoginView.as_view(
             template_name='login.html'      # имя шаблона ниже
         ), 
         name='login'
    ),
    path('accounts/logout/', 
         auth_views.LogoutView.as_view(
             next_page='/accounts/login/'     # после logout вернёт на страницу входа
         ), 
         name='logout'
    ),
    # Можно подключить и другие URL'ы
]

