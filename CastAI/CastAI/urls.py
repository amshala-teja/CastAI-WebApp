"""
URL configuration for CastAI project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from rest_framework.routers import DefaultRouter

router = DefaultRouter()

urlpatterns = [
	path('solarclient/', include('SolarAI.urls')),
	path('', include(router.urls)),
	path('admin/', admin.site.urls),
	path('auth/', include('loginAuth.urls')),  # Root-based login endpoint
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# import pprint
# pprint.pprint(router.urls)

 # Add Login url and
# 1. Login Url should be on root end point.
# 	2. Add a Table to db for user login details.
# 	3. React app should talk to this endpoint for authentication.
# 	4. Add view for the login url
# 	5. This view should handle all the authentication