import base64
import os.path
from django.core.files.base import ContentFile
from io import BytesIO
from .models import SolarPredictionImage
from django.conf import settings
from django.core.files.storage import FileSystemStorage, default_storage
from django.http import HttpResponse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import PySAM.PySSC as pssc

# Assuming lat_lon_df is the same DataFrame from the first code block
# Make sure to load it as done in the first code block
# lat_lon_df = pd.read_csv(r'C:\path\to\your\csv.csv')

ssc = pssc.PySSC()

# Resource inputs for SAM model using df_year:
def set_sam_inputs(df_year, lat, lon, timezone, elevation):
    wfd = ssc.data_create()
    ssc.data_set_number(wfd, b'lat', lat)
    ssc.data_set_number(wfd, b'lon', lon)
    ssc.data_set_number(wfd, b'tz', timezone)
    ssc.data_set_number(wfd, b'elev', elevation)
    ssc.data_set_array(wfd, b'year', df_year.index.year)
    ssc.data_set_array(wfd, b'month', df_year.index.month)
    ssc.data_set_array(wfd, b'day', df_year.index.day)
    ssc.data_set_array(wfd, b'hour', df_year.index.hour)
    ssc.data_set_array(wfd, b'minute', df_year.index.minute)
    ssc.data_set_array(wfd, b'dn', df_year['DNI'])
    ssc.data_set_array(wfd, b'df', df_year['DHI'])
    ssc.data_set_array(wfd, b'wspd', df_year['Wind Speed'])
    ssc.data_set_array(wfd, b'tdry', df_year['Temperature'])
    return wfd 

# def set_system_config(dat):
#     system_capacity = 50
#     ssc.data_set_number(dat, b'system_capacity', system_capacity)
#     ssc.data_set_number(dat, b'dc_ac_ratio', 1.1)
#     ssc.data_set_number(dat, b'tilt', 25)
#     ssc.data_set_number(dat, b'azimuth', 180)
#     ssc.data_set_number(dat, b'inv_eff', 96)
#     ssc.data_set_number(dat, b'losses', 14.0757)
#     ssc.data_set_number(dat, b'array_type', 0)
#     ssc.data_set_number(dat, b'gcr', 0.4)
#     ssc.data_set_number(dat, b'adjust:constant', 0)

def set_system_config(dat, solar_data):
    system_capacity = float(solar_data.get('size', 50))
    dc_ac_ratio = float(solar_data.get('systemRatio', 1.1))
    tilt = float(solar_data.get('tilt', 25))
    azimuth = float(solar_data.get('azimuthAngle', 180))
    inv_eff = float(solar_data.get('inverterEfficiency', 96))
    losses = float(solar_data.get('losses', 14.0757))
    array_type = int(solar_data.get('arrayTpe', 0))
    gcr = float(solar_data.get('gcr', 0.4))
    adjust_constant = float(solar_data.get('adjustConstant', 0))

    ssc.data_set_number(dat, b'system_capacity', system_capacity)
    ssc.data_set_number(dat, b'dc_ac_ratio', dc_ac_ratio)
    ssc.data_set_number(dat, b'tilt', tilt)
    ssc.data_set_number(dat, b'azimuth', azimuth)
    ssc.data_set_number(dat, b'inv_eff', inv_eff)
    ssc.data_set_number(dat, b'losses', losses)
    ssc.data_set_number(dat, b'array_type', array_type)
    ssc.data_set_number(dat, b'gcr', gcr)
    ssc.data_set_number(dat, b'adjust:constant', adjust_constant)


# Global parameters
api_key = 'mq2KJTptRJdO1H5JMsPVjE61pRXcb1McPomaQyqd'
attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle,wind_speed,wind_direction,dew_point,relative_humidity,surface_albedo,cloud_type'
leap_year = 'false'
interval = '60'  # Define interval here
utc = 'false'
your_name = 'Ashish+Sedai'
reason_for_use = 'research'
your_affiliation = 'TTU'
your_email = 'ashis2sedai@gmail.com'
mailing_list = 'true'

# Function to build URL for fetching data
def build_url(lat, lon, year):
    return ('https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.csv?wkt=POINT({lon}%20{lat})'
            '&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}'
            '&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'
            .format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, 
                    email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, 
                    reason=reason_for_use, api=api_key, attr=attributes))

def run_random_forest_model(lat_lon_df,solar_data):
    ssc = pssc.PySSC()

    # Initialize the metrics dataframe
    metrics_df = pd.DataFrame(columns=['Latitude', 'Longitude', 'RMSE', 'MAE', 'R2', 'MAPE', 'MBE'])

    # Loop over each location
    for index, row in lat_lon_df.iterrows():
        lat, lon = row['lat'], row['lon']
        data_for_location = pd.DataFrame()

        for year in range(2015, 2020):
            url = build_url(lat, lon, year)  # Assume build_url is defined elsewhere
            info = pd.read_csv(url, nrows=1)
            timezone, elevation = info['Local Time Zone'], info['Elevation']
            df = pd.read_csv(url, skiprows=2)
            df = df.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq='60Min', periods=525600//60))

            df_year = df[df.index.year == year]
            wfd = set_sam_inputs(df_year, lat, lon, timezone, elevation)  # Assume set_sam_inputs is defined elsewhere

            dat = ssc.data_create()
            set_system_config(dat,solar_data)  # Assume set_system_config is defined elsewhere
            ssc.data_set_table(dat, b'solar_resource_data', wfd)
            ssc.data_free(wfd)

            mod = ssc.module_create(b'pvwattsv5')
            ssc.module_exec(mod, dat)

            generation = np.array(ssc.data_get_array(dat, b'gen'))
            df_year['generation'] = generation * 1000

            ssc.data_free(dat)
            ssc.module_free(mod)

            data_for_location = pd.concat([data_for_location, df_year], axis=0)

        df = data_for_location.copy()
        df.index = pd.to_datetime(df.index)

        # Split data
        total_rows = len(df)
        val_start = int(total_rows * 0.8) - 72 #same thing like below
        test_start = total_rows - 72  #change it to 72 instead of 24 for three days

        X_train = df.iloc[:val_start, df.columns != 'generation']
        y_train = df.iloc[:val_start]['generation']
        X_val = df.iloc[val_start:test_start, df.columns != 'generation']
        y_val = df.iloc[val_start:test_start]['generation']
        X_test = df.iloc[test_start:, df.columns != 'generation']
        y_test = df.iloc[test_start:]['generation']

        # Random fluctuation and scaling
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        random_fluctuation = 1 + np.random.uniform(-0.2, 0.2, X_test.shape)
        X_test_randomized = X_test * random_fluctuation

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test_randomized)

        # Train model
        model = RandomForestRegressor()
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mbe = np.mean(y_test - predictions)
        non_zero_mask = y_test != 0
        mape = np.mean(np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask])) * 100

        # Append metrics
        new_row = pd.DataFrame({
            'Latitude': [lat],
            'Longitude': [lon],
            'RMSE': [rmse],
            'MAE': [mae],
            'R2': [r2],
            'MBE': [mbe],
            'MAPE': [mape]
        })
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

        # Generate plot
        matplotlib.use('Agg') #macos specific fix to not crash on matplotlib gui startup
        plt.figure(figsize=(10, 5))
        hours = np.arange(1, len(y_test) + 1)
        plt.plot(hours, y_test, label='Actual', color='red', linewidth=2)
        plt.plot(hours, predictions, label='Predicted', color='blue', linewidth=2)
        plt.title(f'Solar Power Prediction for Latitude: {lat}, Longitude: {lon}')
        plt.xlabel('Hours')
        plt.ylabel('Solar Power (kW)')
        plt.legend()
        # plt.xticks(hours)
        plt.xticks(np.arange(0, len(hours) + 1, 10))
        plt.tight_layout()

        #  Save Plot
        # plt.figure(figsize=(10, 5))
        # plt.plot(y_test.values, label='Actual', color='red')
        # plt.plot(predictions, label='Predicted', color='blue')
        # plt.title(f'Solar Power Prediction for {lat}, {lon}')
        # plt.legend()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Save plot to DB
        image_instance = SolarPredictionImage()
        image_instance.image.save(f'solar_plot_{lat}_{lon}.png', ContentFile(buffer.read()), save=True)
        plt.close()

        # Return ID & Metrics
        return image_instance.id, metrics_df, hours, y_test, predictions

       # # Generate plot
       #  matplotlib.use('Agg') #macos specific fix to not crash on matplotlib gui startup
       #  plt.figure(figsize=(10, 5))
       #  hours = np.arange(1, 25)
       #  plt.plot(hours, y_test, label='Actual', color='red', linewidth=2)
       #  plt.plot(hours, predictions, label='Predicted', color='blue', linewidth=2)
       #  plt.title(f'Solar Power Prediction for Latitude: {lat}, Longitude: {lon}')
       #  plt.xlabel('Hours')
       #  plt.ylabel('Solar Power (kW)')
       #  plt.legend()
       #  plt.xticks(hours)
       #  plt.tight_layout()
       #  base_path = settings.MEDIA_ROOT
       #  file_name = 'my_plot.png'
       #  file_path = os.path.join(base_path, file_name)
       #  plt.savefig(file_path, format='png' )
       #  return file_path, metrics_df, hours, y_test, predictions


    # save the plot to db, save the plot image to db
    #   1. return the primary key for the img save to db as file path
        #  Save Plot




