from django.urls import path
from .views import SolarPredictionView, SolarPredictionServeImage

urlpatterns = [
    path('generate-prediction/', SolarPredictionView.as_view(), name='prediction-generate-view'),
    path('image/',SolarPredictionServeImage.as_view(),name='serve_image')
]
