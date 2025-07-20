from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from .models import SolarClient 

class BlogPostTests(APITestCase):
    def test_create_blogpost(self):
        url = reverse('api/solarClient-list')
        data = {'title': 'Test Post', 'content': 'Test content'}
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(SolarClient.objects.count(), 1)
        self.assertEqual(SolarClient.objects.get().title, 'Test Post')