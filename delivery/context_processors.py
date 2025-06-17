# delivery/context_processors.py
from django.urls import get_resolver, reverse, NoReverseMatch

def available_urls(request):
    """
    Возвращает список именованных URL, доступных через reverse
    """
    resolver = get_resolver()
    links = []
    # reverse_dict: ключи — имена URL (строки и т.п.), значения — данные для обратного разрешения
    for name in resolver.reverse_dict.keys():
        if not isinstance(name, str):
            continue
        try:
            url = reverse(name)
            links.append({'name': name, 'url': url})
        except NoReverseMatch:
            # пропустим те, которые требуют аргументы
            continue
        
            
        
    return {'available_urls': links}    
