import csv
import os
from datetime import time
from decimal import Decimal

from django.core.management.base import BaseCommand, CommandError
from delivery.models import Order
from courier_service.delivery.services.clustering_solveVRPTW import geocode

class Command(BaseCommand):
    help = "Импортирует заказы из CSV-файла"

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help="Путь к CSV-файлу с данными.")

    def handle(self, *args, **options):
        csv_file = options['csv_file']
        if not os.path.exists(csv_file):
            raise CommandError(f"Файл {csv_file} не найден.")

        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            count = 0
            for row in reader:
                # Убираем пробелы с ключей и значений
                row = {key.strip(): (value.strip() if value is not None else "") for key, value in row.items()}

                address = row['address']
                try:
                    start_hour = int(row['delivery_start'])
                except ValueError:
                    self.stdout.write(self.style.ERROR(
                        f"Ошибка преобразования времени доставки: {row['delivery_start']}"))
                    continue

                delivery_start = time(hour=start_hour, minute=0)
                delivery_end = time(hour=(start_hour + 2) % 24, minute=0)
                contact = row['contact']

                lat,lon = geocode(address)

                try:
                    weight = Decimal(row['weight'])
                except Exception:
                    self.stdout.write(self.style.ERROR(
                        f"Ошибка преобразования веса: {row['weight']}"))
                    continue

                try:
                    volume = Decimal(row['volume'])
                except Exception:
                    self.stdout.write(self.style.ERROR(
                        f"Ошибка преобразования объема: {row['volume']}"))
                    continue

                comment = row['comment']

                order = Order(
                    address=address,
                    delivery_start=delivery_start,
                    delivery_end=delivery_end,
                    contact=contact,
                    weight=weight,
                    volume=volume,
                    comment=comment,
                    lat=lat,
                    lon=lon,
                )
                order.save()
                count += 1

            self.stdout.write(self.style.SUCCESS(f"Импортировано заказов: {count}"))
