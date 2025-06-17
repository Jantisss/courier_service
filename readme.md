# Courier Service VRP System

Автоматизированная система распределения и маршрутизации заказов для курьерской службы, реализованная на базе Django и OR-Tools.

## Возможности

* Импорт заказов из CSV с валидацией и геокодированием адресов
* Кластеризация заказов с учётом временных окон (K-Medoids)
* Решение задачи VRPTW (Vehicle Routing Problem with Time Windows) на базе Google OR-Tools
* Генерация маршрутных листов и их просмотр курьерами через веб-интерфейс
* Отметка статусов доставки и обработка ошибок
* Назначение курьеров на маршруты
* Сводные отчёты.

## Структура проекта

```bash\mcourier_service/ # корень проекта
├── courier_service/            # настройки Django-проекта
│   ├── settings.py             # конфигурация базы, приложений, безопасности
│   └── urls.py                 # глобальные маршруты
├── delivery/                   # основное приложение
│   ├── clustering.py          # алгоритмы кластеризации и VRPTW
│   ├── context_processors.py  # контекст-процессоры Django
│   ├── forms.py               # формы для CSV-импорта и фильтров
│   ├── management/commands/   # кастомные Django-команды
│   │   ├── import_csv.py      # пакетный импорт заказов
│   │   └── assign_orders.py   # назначение заказов курьерам
│   ├── migrations/            # миграции моделей
│   ├── models.py              # модели Courier, Order, Route, RouteItem
│   ├── serializers.py         # DRF-сериализаторы
│   ├── urls.py                # маршруты API и страниц
│   ├── utils.py               # вспомогательные функции маршрутизации
│   ├── views.py               # ViewSet’ы и APIViews
│   └── middleware.py          # Django middleware
├── manage.py                  # управляющая утилита Django
├── requirements.txt           # зависимости проекта
└── README.md                  # этот файл
```

## Установка и запуск

1. Клонировать репозиторий:

```bash

git clone [https://github.com/your-org/courier\_service.git](https://github.com/your-org/courier_service.git)
cd courier\_service

```
2. Создать виртуальное окружение и установить зависимости:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Настроить переменные окружения в файле `.env`:

```ini
DATABASE\_URL=postgres\://user\:password\@localhost:5432/courier\_db
OSRM\_URL=[http://localhost:5000](http://localhost:5000)
SECRET\_KEY=your Django secret key
```
4. Выполнить миграции и создать супер­пользователя:
```bash
python manage.py migrate
python manage.py createsuperuser
```
5. Запустить сервер разработки:

```bash
python manage.py runserver
```

Теперь приложение доступно по адресу `http://127.0.0.1:8000/`.

## Использование
- **Импорт заказов:** в интерфейсе или через команду:
```bash
  python manage.py import_csv path/to/orders.csv
```

* **Геокодирование:** автоматически при импорте
* **Кластеризация и маршрутизация:** нажать кнопку «Сгенерировать маршруты» в веб-интерфейсе логиста
* **Назначение курьеров:** через интерфейс логиста или команду:

```bash
  python manage.py assign_orders
```
* **Просмотр маршрута:** курьер входит в свой аккаунт и открывает страницу маршрутов
* **Отметка доставки:** через веб-интерфейс или API `/route_item/<id>/deliver/`
* **Отчёты:** экспортировать в Excel в разделе «Отчёты»

## Commands management

| Команда         | Описание                                   |
| --------------- | ------------------------------------------ |
| `import_csv`    | Импорт заказов из CSV с геокодированием    |
| `assign_orders` | Автоматическое назначение заказов курьерам |

## Вклад в проект

Для предложений и исправлений создавайте Pull Request в основной репозиторий.

---

*Автор: Старецв Данил Васильевич.*
