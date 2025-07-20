from django.db import models

class SolarClient(models.Model):
    latitude = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)

class BinaryImageModel(models.Model):
    name = models.CharField(max_length=255)
    image_data = models.ImageField(upload_to='plots/')
    created_at = models.DateTimeField(auto_now_add=True)

class SolarPredictionImage(models.Model):
    image = models.ImageField(upload_to='solar_plots/')
    created_at = models.DateTimeField(auto_now_add=True)