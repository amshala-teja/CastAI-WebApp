from rest_framework import serializers
from .models import SolarClient
from decimal import Decimal

class SolarClientCreateSerializer(serializers.ModelSerializer):
	latitude = serializers.FloatField()
	longitude = serializers.FloatField()
	class Meta:
		model = SolarClient
		fields = '__all__'
# class SolarClientCreateSerializer(serializers.ModelSerializer):
#     latitude = serializers.DecimalField(max_digits=9, decimal_places=6)
#     longitude = serializers.DecimalField(max_digits=9, decimal_places=6)
#
#     def to_internal_value(self, data):
#         data['latitude'] = Decimal(data.get('latitude'))
#         data['longitude'] = Decimal(data.get('longitude'))
#         return super().to_internal_value(data)
#
#     class Meta:
#         model = SolarClient
#         fields = '__all__'
