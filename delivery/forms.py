from django import forms
from django.forms import modelformset_factory
from .models import Courier, RouteItem 
from django.contrib.auth.models import User, Group



class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(
        label="Выберите CSV файл",
        help_text="Файл должен содержать заголовки столбцов"
    )

class CSVImportForm(forms.Form):
    csv_file = forms.FileField(
        label="CSV файл с заказами",
        help_text="Ожидает CSV с любыми заголовками; дальше вы укажете маппинг"
    )

class CSVMappingForm(forms.Form):
    address         = forms.ChoiceField(label="Адрес",         required=False)
    delivery_start  = forms.ChoiceField(label="Начало интервала",  required=False)
    delivery_end    = forms.ChoiceField(label="Конец интервала",   required=False)
    weight          = forms.ChoiceField(label="Вес",           required=False)
    volume          = forms.ChoiceField(label="Объём",         required=False)
    contact         = forms.ChoiceField(label="Контакт",       required=False)
    comment         = forms.ChoiceField(label="Комментарий",   required=False)

    def __init__(self, *args, headers=None, initial=None, **kwargs):
        """
        headers — список строк из CSV-заголовков,
        initial — предварительный маппинг {поле: header или ''}.
        """
        super().__init__(*args, initial=initial, **kwargs)
        headers = headers or []
        choices = [('', '---')] + [(h, h) for h in headers]
        for field in self.fields.values():
            field.choices = choices

class CourierForm(forms.ModelForm):
    username  = forms.CharField(label='Логин', max_length=150)
    password1 = forms.CharField(
        label='Пароль',
        widget=forms.PasswordInput,
        required=False,   # при редактировании необязательно
    )
    password2 = forms.CharField(
        label='Подтверждение пароля',
        widget=forms.PasswordInput,
        required=False,
    )

    class Meta:
        model = Courier
        fields = [
            'fio',
            'contact',
            'vehicle_capacity_weight',
            'vehicle_capacity_volume',
            'is_available',
        ]
        labels = {
            'fio': 'ФИО курьера',
            'contact': 'Контактный телефон',
            'vehicle_capacity_weight': 'Грузоподъёмность (кг)',
            'vehicle_capacity_volume': 'Объём багажника (м³)',
            'is_available': 'Доступен для доставки',
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # при редактировании подставляем текущий логин
        if self.instance and self.instance.pk:
            self.fields['username'].initial = self.instance.user.username

    def clean_username(self):
        username = self.cleaned_data['username']
        qs = User.objects.filter(username=username)
        # исключаем себя, если это апдейт
        if self.instance and self.instance.pk:
            qs = qs.exclude(pk=self.instance.user.pk)
        if qs.exists():
            raise forms.ValidationError("Этот логин уже занят.")
        return username

    def clean(self):
        cleaned = super().clean()
        p1 = cleaned.get('password1')
        p2 = cleaned.get('password2')
        # если ввели хотя бы одно поле с паролем — требуем совпадение
        if p1 or p2:
            if p1 != p2:
                self.add_error('password2', "Пароли не совпадают.")
        return cleaned

    def save(self, commit=True):
        courier = super().save(commit=False)

        # Решаем, создаём ли нового User или обновляем существующего
        if self.instance and self.instance.pk and courier.user_id:
            # апдейт существующего
            user = courier.user
            new_login = self.cleaned_data['username']
            if user.username != new_login:
                user.username = new_login
            # меняем пароль только если ввели
            pw = self.cleaned_data.get('password1')
            if pw:
                user.set_password(pw)
        else:
            # создаём нового пользователя
            user = User.objects.create_user(
                username=self.cleaned_data['username'],
                password=self.cleaned_data['password1'] or User.objects.make_random_password()
            )

        user.save()

         # 1) Привязать user к группе courier
        courier_group, _ = Group.objects.get_or_create(name='courier')
        user.groups.add(courier_group)

        # 2) Синхронизировать поля Courier
        courier.user = user
        courier.login = user.username

        if commit:
            courier.save()
        return courier

class RouteItemForm(forms.ModelForm):
    class Meta:
        model  = RouteItem
        fields = ['status', 'courier_comment']
        widgets = {
            'status': forms.Select(attrs={'class': 'status-select'}),
            'courier_comment': forms.Textarea(attrs={'rows': 2, 'cols': 20}),
        }

RouteItemFormSet = modelformset_factory(
    RouteItem,
    form=RouteItemForm,
    extra=0
)