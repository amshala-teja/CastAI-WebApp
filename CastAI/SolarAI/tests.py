import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent.__str__())
print (sys.path)
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "CastAI.settings")

import django
django.setup()

from django.core.management import call_command
from django.urls import reverse

# Generating the URL for custom_post_action
# url_post = reverse('my-viewset-custom-post-action')
# print(url_post)  # Output: '/my-viewset/custom-post-action/'

# Assuming you have a MyObject instance with pk=1
# Generating the URL for custom_get_action with a specific object
url_get = reverse('SolarAI', kwargs={'pk': 1})
print(url_get)
# Output: '/my-viewset/1/custom-get-action/'
