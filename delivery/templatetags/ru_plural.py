from django import template

register = template.Library()

@register.filter
def ru_plural_minute(value):
    """
    Склоняет слово «минута» по-русски: 1 минута, 2–4 минуты, 5+ минут.
    """
    try:
        n = abs(int(value))
    except (ValueError, TypeError):
        return ''
    if n % 10 == 1 and n % 100 != 11:
        return 'минута'
    elif 2 <= n % 10 <= 4 and not (12 <= n % 100 <= 14):
        return 'минуты'
    else:
        return 'минут'
