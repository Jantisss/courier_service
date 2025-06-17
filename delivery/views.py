from rest_framework import viewsets
from django.shortcuts import render,redirect,get_object_or_404
from django.http import JsonResponse
from .models import Order, Courier, Route, RouteItem
from .serializers import OrderSerializer, CourierSerializer
from .services.clustering_solveVRPTW import add_route, geocluster
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import folium
import numpy as np
import requests
import json
from datetime import datetime, time
import pandas as pd
from django.db import transaction
import csv
from decimal import Decimal
from .forms import CSVUploadForm, CSVMappingForm, CSVImportForm
from django.views.generic import ListView, UpdateView, CreateView
from django.urls import reverse_lazy, reverse
from django.db.models import Q,Prefetch
from django.contrib import messages
from django.core.management import call_command
from .services.utils import geocode, orders_to_dataframe
from .forms import CourierForm, RouteItemFormSet
from django.utils import timezone
from django.utils.timezone import make_aware, get_current_timezone
from collections import defaultdict
from django.contrib.auth.decorators import login_required, user_passes_test
from django.conf import settings
from django.contrib.auth.views import LoginView
from django.views.generic import DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.db.models.deletion import ProtectedError
from django import forms


ALPHA = 0
BETA = 10000

def orders_to_dataframe(order_number = None, cluster = None):
    print("ok")
    """
    Преобразует все заказы из БД в pandas DataFrame.
    """
    if order_number:
        qs = Order.objects.filter(order_number=order_number).values(
            'order_number', 'address', 'lat', 'lon',
            'delivery_start', 'delivery_end', 'weight', 'volume', 'contact', 'comment', 'cluster'
        )
    if cluster:
        qs = Order.objects.filter(cluster=cluster).values(
                    'order_number', 'address', 'lat', 'lon',
                    'delivery_start', 'delivery_end', 'weight', 'volume', 'contact', 'comment', 'cluster'
                )
    if not cluster and not order_number:
        qs = Order.objects.all().values(
            'order_number', 'address', 'lat', 'lon',
            'delivery_start', 'delivery_end', 'weight', 'volume', 'contact', 'comment', 'cluster'
        )
    df = pd.DataFrame.from_records(qs)
    
    df['lat'] = df['lat'].astype(float)
    df['lon'] = df['lon'].astype(float)
    df['weight'] = df['weight'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df


def import_orders(request):
    import_form  = CSVUploadForm()
    mapping_form = None
    headers      = []
    errors       = []

    # Шаг 1: загрузка файла и подготовка формы маппинга
    if request.method == 'POST' and 'mapping_submitted' not in request.POST:
        import_form = CSVImportForm(request.POST, request.FILES)
        if import_form.is_valid():
            csv_file = import_form.cleaned_data['csv_file']
            lines    = csv_file.read().decode('utf-8').splitlines()
            headers  = [h.strip() for h in lines[0].split(';')]
            request.session['csv_data'] = lines

            initial = {}
            for fld in ['address','delivery_start','delivery_end','weight','volume','contact','comment']:
                initial[fld] = fld if any(h.lower().startswith(fld) for h in headers) else ''

            mapping_form = CSVMappingForm(headers=headers, initial=initial)
            return render(request, 'delivery/import_orders.html', {
                'mapping_form': mapping_form,
                'errors':       errors,
                'headers':      headers,
            })

    # Шаг 2: импорт с учётом маппинга
    elif request.method == 'POST' and 'mapping_submitted' in request.POST:
        headers      = request.POST.getlist('headers')
        mapping_form = CSVMappingForm(request.POST, headers=headers)

        if mapping_form.is_valid():
            lines  = request.session.get('csv_data', [])
            reader = csv.DictReader(lines, delimiter=';')
            count  = 0

            for idx, row in enumerate(reader, start=1):
                vals = {}
                for fld, hdr in mapping_form.cleaned_data.items():
                    if hdr:
                        vals[fld] = row.get(hdr, '').strip()
                try:
                    lat, lon = geocode(vals.get('address', ''))
                    Order.objects.create(
                        address=vals.get('address',''), lat=lat, lon=lon,
                        delivery_start=vals.get('delivery_start',''),
                        delivery_end=vals.get('delivery_end',''),
                        weight=vals.get('weight',''), volume=vals.get('volume',''),
                        contact=vals.get('contact',''), comment=vals.get('comment','')
                    )
                    count += 1
                except Exception as e:
                    errors.append({'row': idx, 'errors': [str(e)]})

            if errors:
                messages.error(request, f"При импорте возникли ошибки в строках: {[e['row'] for e in errors]}")
            else:
                messages.success(request, f"Импортировано {count} заказов")
                # Очистим данные из сессии
                request.session.pop('csv_data', None)
                # Редирект на окно логиста
                return redirect('delivery:logistics_dashboard')

    # Рендер начальной формы или формы маппинга с ошибками
    return render(request, 'delivery/import_orders.html', {
        'import_form':  import_form,
        'mapping_form': mapping_form,
        'errors':       errors,
        'headers':      request.session.get('csv_data', [''])[0].split(';')
                           if request.session.get('csv_data') else []
    })

def clear_orders(request):
    """
    Вызывает management command clear_orders и возвращает на дашборд логиста.
    """
    if request.method == 'POST':
        # call_command('clear_orders',)
        with transaction.atomic():
            # 1) Сначала удаляем все позиции маршрутов
            RouteItem.objects.all().delete()
            # 2) Потом — сами маршруты
            Route.objects.all().delete()
            # 3) И только затем — заказы
            Order.objects.all().delete()

        messages.success(request, 'Все заказы и связанные маршруты удалены.')

    return redirect('delivery:logistics_dashboard')

class OrderViewSet(viewsets.ModelViewSet):
    queryset = Order.objects.all()
    serializer_class = OrderSerializer

class CourierViewSet(viewsets.ModelViewSet):
    queryset = Courier.objects.all()
    serializer_class = CourierSerializer

class LogisticsDashboardView(ListView):
    model = Order
    template_name = 'delivery/logistics_dashboard.html'
    context_object_name = 'orders'
    paginate_by = 8

    def get_queryset(self):
        qs = (
            super()
            .get_queryset()
            .select_related('courier')
            .prefetch_related('route_items')       
        )
        search = self.request.GET.get('search')
        if search:
            qs = qs.filter(Q(address__icontains=search) | Q(contact__icontains=search))
        cluster = self.request.GET.get('cluster')
        if cluster:
            qs = qs.filter(cluster=cluster)
        courier = self.request.GET.get('courier')
        if courier:
            qs = qs.filter(courier_id=courier)
        return qs
    
    def post(self, request, *args, **kwargs):
        """
        Обрабатываем сохранение статусов/комментариев.
        """
        # Создаём formset на основе всех RouteItem,
        # относящихся к отображаемым Order-ам
        orders_qs = self.get_queryset()
        items_qs = RouteItem.objects.filter(order__in=orders_qs)
        formset = RouteItemFormSet(request.POST, queryset=items_qs)
        if formset.is_valid():
            formset.save()
            # После сохранения — перезагружаем страницу без POST, чтобы избежать повторной отправки
            return redirect(request.path + '?' + request.META.get('QUERY_STRING', ''))
        # Если не валидно — покажем страницу с ошибками
        return self.get(request, *args, formset=formset, **kwargs)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        # Все курьеры для формы выбора
        ctx['couriers'] = Courier.objects.filter(is_available=True)
        # Уникальные кластеры для фильтра
        ctx['clusters_unique'] = Order.objects.values_list('cluster', flat=True).distinct()

        # Подсчёт кластеров и назначенных курьеров
        qs_all = self.get_queryset()
        clusters_count = qs_all.values_list('cluster', flat=True).distinct().count()
        assigned_count = qs_all.values_list('courier_id', flat=True).distinct().exclude(courier_id__isnull=True).count()
        missing_count = max(0, clusters_count - assigned_count)
        cluster = self.request.GET.get('cluster') 
        cluster = cluster if cluster else None
        courier = self.request.GET.get('courier')
        courier = courier if courier else None


        # print("courier", courier)
        # print("cluster", cluster)
        ctx['clusters_count'] = clusters_count
        ctx['assigned_couriers_count'] = assigned_count
        ctx['missing_couriers_count'] = missing_count

        ctx['vcw']   = self.request.GET.get('vehicle_capacity_weight', 620)
        ctx['vcv']   = self.request.GET.get('vehicle_capacity_volume', float(2.54))
        ctx['alpha'] = self.request.GET.get('alpha', 1)
        ctx['beta']  = self.request.GET.get('beta', 25000)

        # ctx['map_html'] = generate_map(cluster_filter=cluster,courier_filter=courier)

        # print("info", ctx['vcw'], ctx['vcv'], ctx['alpha'], ctx['beta'], ctx)
        if 'formset' in kwargs:
            ctx['formset'] = kwargs['formset']
        else:
            orders_qs = self.get_queryset()
            items_qs = RouteItem.objects.filter(order__in=orders_qs)
            ctx['formset'] = RouteItemFormSet(queryset=items_qs)

        return ctx

class OrderUpdateView(UpdateView):
    model = Order
    fields = [
        'address', 'delivery_start', 'delivery_end',
        'weight', 'volume', 'contact', 'comment',
        'cluster', 'courier'
    ]
    template_name = 'delivery/order_form.html'
    success_url = reverse_lazy('delivery:logistics_dashboard')

    def get_form(self, form_class=None):
        form = super().get_form(form_class)
        # доп. поле sequence
        form.fields['sequence'] = forms.IntegerField(
            label='Порядок следования',
            required=False
        )
        # поставить initial из первого RouteItem, если есть
        try:
            route_item = self.object.route_items.first()
            form.fields['sequence'].initial = route_item.sequence
        except AttributeError:
            pass
        return form

    def form_valid(self, form):
        # сначала сохраняем Order
        response = super().form_valid(form)
        seq = form.cleaned_data.get('sequence')
        if seq is not None:
            # обновляем в RouteItem
            route_item = self.object.route_items.first()
            if route_item:
                route_item.sequence = seq
                route_item.save()
        return response
    

def distribute_orders(request):

    vcw   = float(request.GET.get('vehicle_capacity_weight', 625))
    vcv   = float(request.GET.get('vehicle_capacity_volume', 2.54))
    alpha = float(request.GET.get('alpha', 1))
    beta  = float(request.GET.get('beta', 25000))
    
    call_command('assign_orders', vehicle_capacity_weight=vcw,
        vehicle_capacity_volume=vcv,
        alpha=alpha,
        beta=beta)  
    messages.success(request, 'Заказы успешно распределены между курьерами.')
    return redirect('delivery:logistics_dashboard')

def generate_map(cluster_filter = None, courier_filter = None):
    if cluster_filter:
        orders_df = orders_to_dataframe(cluster=cluster_filter)    
        print("orders_df_generate_map_cluster", orders_df)
        
    else:
        try:
            orders_df = orders_to_dataframe()  
        except:
            return
        if orders_df.empty:
            return  
        print("orders_df_generate_map", orders_df)

    center_lat = orders_df['lat'].mean()
    center_lon = orders_df['lon'].mean()
    map_obj = folium.Map(location=[center_lat, center_lon], 
                         zoom_start=12,
                         width='100%',
                         height='70%')

    unique_clusters = sorted(orders_df['cluster'].unique())
    num_clusters = len(unique_clusters)
    cmap_instance = cm.get_cmap('tab20', num_clusters)
    colors = [mcolors.rgb2hex(cmap_instance(i)) for i in range(num_clusters)]
    cluster_to_color = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

    print("\nЦветовая палитра кластеров:")
    for cluster, color in cluster_to_color.items():
        print(f"Кластер {cluster}: {color}")

    for idx, row in orders_df.iterrows():
        cluster = row['cluster']
        color = cluster_to_color.get(cluster, 'gray')
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"Order: {row['order_number']}\nStart: {row['delivery_start']}\nКластер: {cluster}"
        ).add_to(map_obj)
    
    if courier_filter:
        items_RouteItem_qs = (
            RouteItem.objects
                .filter(route__courier=courier_filter)
                .select_related('order', 'route')
                .order_by('route__cluster', 'sequence')
        )
    if cluster_filter:
        items_RouteItem_qs = (
            RouteItem.objects
                .filter(route__cluster=cluster_filter)
                .select_related('order', 'route')
                .order_by('route__cluster', 'sequence')
        )
    if not cluster_filter and not courier_filter:
        items_RouteItem_qs = (
            RouteItem.objects
                .all()
                .select_related('order', 'route')
                .order_by('route__cluster', 'sequence')
        )
    print("items_RouteItem_qs", items_RouteItem_qs)

    # Создание словаря cluster:[numbers_orders]
    routes = {}
    for item in items_RouteItem_qs:
        # print("item",item.route.cluster)
        if routes.get(item.route.cluster):
            routes[item.route.cluster].append(item.order.order_number)
        else:
            routes[item.route.cluster] = [item.order.order_number]
        
        # routes.update({item.route.cluster:})
    print('routes', routes, cluster_filter,courier_filter)
    for cluster, route in routes.items():
        if route is None:
            continue
        cluster_df = orders_df[orders_df['cluster'] == cluster].reset_index(drop=True)
        # Если route - кортеж, обрабатываем по частям
        add_route(route, cluster, cluster_to_color, map_obj, cluster_df)

    map_html = map_obj._repr_html_()

    return(map_html)

def cluster_only(request):
    # 1. Параметры из GET
    try:
        vcv   = float(request.GET.get('vehicle_capacity_volume', 100))
        vcw   = float(request.GET.get('vehicle_capacity_weight', 100))
        alpha = float(request.GET.get('alpha', 1))
        beta  = float(request.GET.get('beta', 25000)) 
    except ValueError:
        messages.error(request, "Неверные параметры кластеризации")
        return redirect('delivery:logistics_dashboard')

    # 2. Готовим данные
    orders_qs = Order.objects.all()
    if not orders_qs.exists():
        messages.info(request, "Нет заказов для кластеризации")
        return redirect('delivery:logistics_dashboard')

    data = [(
        o.order_number,
        float(o.lat), float(o.lon),
        o.delivery_start.hour * 60, o.delivery_end.hour * 60,
        float(o.weight), float(o.volume)
    ) for o in orders_qs]

    # 3. Число кластеров — например, по числу курьеров, либо любая другая логика
    n_clusters = Courier.objects.filter(is_available=True).count() or 1

    # 4. Собственно кластеризация
    clusters_map = geocluster(
        data,
        n_clusters=n_clusters,
        ALPHA=alpha,
        BETA=beta,
        vcw=vcw,
        vcv=vcv
    )
    # clusters_map — dict: {order_number: cluster_label, …}

    # 5. Сохраняем результат в поле order.cluster
    for order_number, cluster_label in clusters_map.items():
        Order.objects.filter(order_number=order_number).update(cluster=cluster_label)

    messages.success(request, "Кластеризация выполнена ({} кластеров)".format(n_clusters))
    return redirect('delivery:logistics_dashboard')

# CRUD для курьеров
class CourierListView(ListView):
    model = Courier
    template_name = 'delivery/courier_list.html'
    context_object_name = 'couriers'

class CourierDeleteView(DeleteView):
    model = Courier
    template_name = 'delivery/courier_list.html'
    context_object_name = 'couriers'

class CourierCreateView(CreateView):
    model = Courier
    form_class = CourierForm
    template_name = 'delivery/courier_form.html'
    success_url = reverse_lazy('delivery:courier_list')

class CourierUpdateView(UpdateView):
    model = Courier
    form_class = CourierForm
    template_name = 'delivery/courier_form.html'
    success_url = reverse_lazy('delivery:courier_list')

def route_sheets(request):
    today = timezone.localdate()

    # 1) Выбираем только позиции сегодняшних маршрутов
    items_qs = (
        RouteItem.objects
            # .filter(route__date=today)
            .select_related('order', 'route__courier')
            .order_by('route__cluster', 'sequence')
    )

    # 2) Группируем по курьеру
    by_courier = defaultdict(list)
    for item in items_qs:
        by_courier[item.route.courier].append(item)

    summaries = []
    tz = get_current_timezone()
    for courier, items in by_courier.items():
        # время смены
        starts = [i.order.delivery_start for i in items]
        ends   = [i.order.delivery_end   for i in items]
        shift_start = min(starts)
        shift_end   = max(ends)

        # статусы
        delivered = [i for i in items if i.status == 'delivered']
        delayed   = [i for i in items if i.status == 'late']
        undel     = [i for i in items if i.status == 'not_delivered']

        # среднее время доставки
        avg_minutes = 0
        if delivered:
            # находим самый ранний delivered_at
            first_dt = min(i.delivered_at for i in delivered)
            # суммируем интервалы от первой доставки до каждой последующей
            total_seconds = sum(
                (i.delivered_at - first_dt).total_seconds()
                for i in delivered
            )
            # переводим в минуты и усредняем
            avg_minutes = int(total_seconds / 60 / len(delivered))

        # замечания
        remarks = [
            f"Требуется повторная попытка по адресу №{idx}, {i.order.address}"
            for idx, i in enumerate(items, 1)
            if i.status == 'not_delivered'
        ]
        
        summaries.append({
            'courier': courier,
            # 'cluster': items[0].order.cluster,
            'items': items,
            'shift_start': shift_start,
            'shift_end': shift_end,
            'delivered_count': len(delivered),
            'delayed_count':   len(delayed),
            'undelivered_count': len(undel),
            'avg_minutes': avg_minutes,
            'remarks': remarks,
        })

    return render(request, 'delivery/route_sheets.html', {
        'today': today,
        'summaries': summaries,
    })  


def route_sheet_for_courier(request, courier_id):
    today   = timezone.localdate()
    # courier = get_object_or_404(Courier, pk=courier_id)

    try:
        courier = Courier.objects.get(pk=courier_id)
    except Courier.DoesNotExist:
        return render(request, 'delivery/route_not_generated.html')

    # все позиции маршрута этого курьера на сегодня
    items_qs = (
        RouteItem.objects
                 .filter(route__courier=courier)
                 .select_related('order', 'route')
                 .order_by('route__cluster', 'sequence')
    )

    if not items_qs:
        return render(request, 'delivery/route_not_generated.html')

    # создаём formset
    if request.method == 'POST':
        formset = RouteItemFormSet(request.POST, queryset=items_qs)
        if formset.is_valid():
            for inst in formset.save(commit=False):
                if (inst.status == 'delivered' or inst.status == 'late') and not inst.delivered_at:
                    inst.delivered_at = timezone.now()
                inst.save()
            return redirect('delivery:route_sheet_for_courier', courier_id=courier_id)
    else:
        formset = RouteItemFormSet(queryset=items_qs)

    # zip-пары (item, form), форму того же порядка, что queryset
    paired = list(zip(items_qs, formset.forms))

    # группируем по кластеру
    clusters: dict[int, list[dict]] = {}
    clusters_list = []
    for item, form in paired:
        clusters_list.append(item.route.cluster)
        clusters.setdefault(item.route.cluster, []).append({
            'item': item,
            'form': form
        })
    print("test_clusters", clusters_list)
    map_html = generate_map(cluster_filter=clusters_list[0], courier_filter=courier_id)

    return render(request, 'delivery/route_sheet.html', {
        'courier': courier,
        'today': today,
        'clusters': clusters,   # {cluster_num: [ {'item':…, 'form':…}, … ]}
        'formset': formset,
        'map_html': map_html
    })



class CourierDeleteView(
    LoginRequiredMixin,
    UserPassesTestMixin,
    DeleteView
):
    model = Courier
    template_name = 'delivery/courier_confirm_delete.html'
    context_object_name = 'courier'
    success_url = reverse_lazy('delivery:courier_list')
    login_url = reverse_lazy('login')

    def test_func(self):
        user = self.request.user
        return user.is_superuser or user.groups.filter(name='logist').exists()

    def post(self, request, *args, **kwargs):
        """
        Переопределяем post, чтобы поймать ProtectedError при on_delete=PROTECT
        """
        self.object = self.get_object()
        try:
            # super().post вызывает внутри delete()
            return super().post(request, *args, **kwargs)
        except ProtectedError:
            # Рендерим тот же шаблон с ошибкой
            return render(
                request,
                self.template_name,
                {
                    'courier': self.object,
                    'error': 'Нельзя удалить курьера: существуют связанные маршруты.'
                },
                status=400
            )
        
class CourierLoginView(LoginView):
    template_name = 'login.html'
    

    def form_invalid(self, form):
        # Этот метод вызывается, если форма аутентификации не прошла валидацию
        messages.error(self.request, "Неправильный логин или пароль")
        return super().form_invalid(form)

    def get_success_url(self):
        user = self.request.user
        # Если курьер — сразу на его маршрут
        if user.groups.filter(name='courier').exists():
            courier = Courier.objects.get(user_id=user.id)
            return reverse(
                'delivery:route_sheet_for_courier',
                kwargs={'courier_id': courier.id}
            )
        # Иначе — стандартный next или LOGIN_REDIRECT_URL
        return super().get_success_url()