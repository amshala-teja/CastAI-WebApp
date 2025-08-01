# Generated by Django 4.2.17 on 2025-03-10 21:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("SolarAI", "0002_binaryimagemodel"),
    ]

    operations = [
        migrations.CreateModel(
            name="SolarPredictionImage",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("image", models.ImageField(upload_to="solar_plots/")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
