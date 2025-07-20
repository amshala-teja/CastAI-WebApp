# from django.core.files.storage import FileSystemStorage
# from django.http import FileResponse, HttpResponse
# from .serializers import SolarClientCreateSerializer
# from rest_framework import status
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from .utils import run_random_forest_model
# import pandas as pd
# 
# 
# 
# class SolarPredictionView(APIView):
#     def post(self, request, *args, **kwargs):
#         print(f"Incoming request data: {request.data}")  # Debug: Log incoming request data
#         serializer = SolarClientCreateSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             lat_lon_df = pd.DataFrame([{'lat': request.data['latitude'], 'lon': request.data['longitude']}])
#             figdata_path, metrics_df, hours, y_test, predictions = run_random_forest_model(lat_lon_df)
#             # Generate a unique key for the image file
#             unique_image_key = get_random_string(12) + ".png"
# 
#             # Save the generated image to media storage
#             fs = FileSystemStorage()
#             with open(figdata_path, 'rb') as f:
#                 fs.save(unique_image_key, f)
# 
#             # Prepare the response with the unique image key
#             response = {
#                 'serializer_data': serializer.data,
#                 'image_key': unique_image_key,  # Updated to include the unique image key
#                 'metrics': metrics_df.to_dict(),
#                 'hours': hours,
#                 'y_test': y_test.tolist(),
#                 'predictions': predictions.tolist()
#             }
#             return Response(data=response, status=status.HTTP_201_CREATED)
#         else:
#             return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#         
# # for post method, add image generation after running the random forest model
# 
# # for post method,
#     # 1. add image generation after running the random forest model
#         # B. WHAT IS figdata_path
#         # The figdata_path variable is mentioned in the previous code but is not used or needed anymore.Instead, we now
#         # directly        generate and save
#         # the        image in the
#         # database        with a unique name, and the image is stored in the SolarPredictionImage model.
#         # In        the SolarPredictionView(the
#         # post        method), the
#         # image is saved        to
#         # the        database        without        the        need        for figdata_path
#     # 2. double check random forest model, what is input and what is output?
#     #3. save the image generated to the db with a unique key/token
# 
# class SolarPredictionServeImage(APIView):
#     def get(self,request):
#         image_key = request.GET.get('image_key')  # Fetch image key from query params
#         if not image_key:
#             return HttpResponse('Image key is required.', status=400)
# 
#         fs = FileSystemStorage()
#         if not fs.exists(image_key):
#             return HttpResponse('Image not found.', status=404)
# 
#         with fs.open(image_key, 'rb') as image_file:
#             response = HttpResponse(image_file.read(), content_type='image/png')
#             response['Content-Disposition'] = f'inline; filename="{image_key}"'
#             return response


# 1. Input should have image key that we added above
# 2. Image should be pulled from db and sent as image response
#
import os
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse, HttpResponse
from django.utils.crypto import get_random_string
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from .utils import run_random_forest_model
from .models import SolarPredictionImage
from .serializers import SolarClientCreateSerializer
import pandas as pd


class SolarPredictionView(APIView):
    def post(self, request, *args, **kwargs):
        print(f"Incoming request data: {request.data}")  # Debug: Log incoming request data
        serializer = SolarClientCreateSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            lat_lon_df = pd.DataFrame([{'lat': request.data['latitude'], 'lon': request.data['longitude']}])

            # Run the Random Forest Model and get the generated image
            image_instance_id, metrics_df, hours, y_test, predictions = run_random_forest_model(lat_lon_df, request.data['solarData'])

            # Generate a unique key for the image file
            unique_image_key = get_random_string(12) + ".png"

            # Save the image to the database
            image_instance = SolarPredictionImage.objects.get(id=image_instance_id)
            # image_instance.image.name = f"solar_plot_{unique_image_key}"
            # image_instance.save()
            # Open the plot image and save it to the model using the unique key
            with open(image_instance.image.path, 'rb') as f:
                image_instance.image.save(f"{unique_image_key}", f, save=True)

            # Log the linked image_instance_id for debugging
            print(f"Image saved with instance ID: {image_instance_id}, file name: {image_instance.image.name}")

            # Prepare the response with the unique image key
            response = {
                'serializer_data': serializer.data,
                'image_instance_id': image_instance_id,  # Include the image instance ID
                'image_key': unique_image_key,  # Include the unique image key
                'metrics': metrics_df.to_dict(),
                'hours': hours.tolist(),
                'y_test': y_test.tolist(),
                'predictions': predictions.tolist()
            }
            return Response(data=response, status=status.HTTP_201_CREATED)
        else:
            print(f"Serializer errors: {serializer.errors}")  # Log validation errors
            return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SolarPredictionServeImage(APIView):
    def get(self, request):
        image_key = request.GET.get('image_key')  # Fetch image key from query params
        if not image_key:
            return HttpResponse('Image key is required.', status=400)

        # Fetch the image from the database
        try:
            image_instance = SolarPredictionImage.objects.get(image__contains=image_key)
        except SolarPredictionImage.DoesNotExist:
            return HttpResponse('Image not found.', status=404)

        # Check if the file actually exists before reading
        image_file_path = image_instance.image.path
        if not os.path.exists(image_file_path):
            return HttpResponse('Image file missing on server.', status=500)
        # Serve the image from the database
        # image_file_path = image_instance.image.path
        with open(image_file_path, 'rb') as image_file:
            response = HttpResponse(image_file.read(), content_type='image/png')
            response['Content-Disposition'] = f'inline; filename="{image_key}"'
            return response
