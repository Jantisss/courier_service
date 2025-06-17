from django.conf import settings
from django.http import HttpResponseForbidden
from django.urls import reverse,resolve
from django.shortcuts import render
from delivery.models import Courier



# app/middleware.py

class CourierOnlyMiddleware:
    """
    Блокирует всем курьерам доступ ко всем URL,
    кроме их собственной страницы маршрутного листа, 
    а также login/logout.
    """
    def __init__(self, get_response):
        self.get_response = get_response
        # урлы, которые можно любому курьеру
        self.whitelist = {
            settings.LOGIN_URL,
            reverse('logout'),
        }

    def __call__(self, request):
        user = request.user
        
        print('ok', user.is_authenticated, user.groups)
        # только для аутентифицированных курьеров применять блокировку
        if user.is_authenticated and user.groups.filter(name='courier').exists():
            courier = Courier.objects.get(user_id = user.id)
            path = request.path_info  # без GET-параметров
            
            # 1) разрешаем login/logout
            if path in self.whitelist:
                
                return self.get_response(request)
            
            # 2) пробуем разобрать URL через resolve()
            try:
                match = resolve(path)
            except Exception:
                # не смогли разобрать — запрещаем
                back_url = reverse(
                   'delivery:route_sheet_for_courier',
                    kwargs={'courier_id': courier.id}
                )
                
                context = {
                    'message': "У вас нет доступа к этому разделу.",
                    'back_url': back_url
                }
                
                response = render(request, '403.html', context)
                response.status_code = 403
                return response
            
            # 3) если это наш view маршрутного листа и id совпадает — разрешаем
            if match.view_name == 'delivery:route_sheet_for_courier':
                # kwargs извлекаются уже из строки URL, вне зависимости от слэша
                cid = match.kwargs.get('courier_id')
                # courier = Courier.objects.get(user_id = user.id)
                if cid is not None and int(cid) == courier.id:
                    return self.get_response(request)
            
            # во всех остальных случаях — запрет
            back_url = reverse(
                   'delivery:route_sheet_for_courier',
                    kwargs={'courier_id': courier.id}
            )
            context = {
                'message': "У вас нет доступа к этому разделу.",
                'back_url': back_url
            }
            response = render(request, '403.html', context)
            response.status_code = 403
            return response
        
        # для остальных (не курьеры) — просто пропускаем запрос дальше
        return self.get_response(request)
