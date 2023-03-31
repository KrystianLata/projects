import requests
import numpy as np
import csv
import pandas as pd
import json
from typing import List
from bs4 import BeautifulSoup
from IPython.display import clear_output
import geopy.distance
# import otodom_scraper
from modules.otodom_scraper import get_subpages_list, get_offers, scrape_offers
pd.set_option('display.max_columns', None)
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
import joblib
import pickle




def add_dummies(df: pd.DataFrame, column_name:List[str]):
    """Transforming columns provided in list to dummies"""
    dummies = pd.get_dummies(df[column_name])
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column_name, axis=1)
    return df

def create_dummies_from_features(df: pd.DataFrame, column_name: str):
    """ Split column with list of values to dummie columns"""
    
    # Split the values in the column into separate categories
    df[column_name] = df[column_name].str.split(", ")
    
    # Create dummie columns for each category
    dummie_columns = pd.get_dummies(df[column_name].apply(pd.Series).stack(), prefix = column_name).sum(level=0)
    
    # Adding new columns to the original df
    df = pd.concat([df, dummie_columns], axis=1)
    df = df.drop(columns = [column_name])
    
    return df

def add_distance_to_point(df: pd.DataFrame, point_address: str, point_name: str):
    """Function adds new column to DataFrame with distance from given point"""
    
    # Get coordinates of point
    geolocator = Nominatim(user_agent='my_application')
    location = geolocator.geocode(point_address)
    if location:
        point_location = (location.latitude, location.longitude)
        # print(f'{point_name} coordinates: {point_location}')
    else:
        raise ValueError('Could not get coordinates for the given address')
    
    # Calculate distances
    distances = []
    for index, row in df.iterrows():
        location_lat, location_lon = row['LATITUDE'], row['LONGITUDE']
        location = (location_lat, location_lon)
        distance = geopy.distance.distance(location, point_location).km
        distance = distance
        distances.append(distance)
    
    # Add new column to DataFrame
    df[f'DISTANCE_FROM_{point_name}_KM'] = distances
    df[f'DISTANCE_FROM_{point_name}_KM'] = df[f'DISTANCE_FROM_{point_name}_KM'].apply(lambda x: '{:.2f}'.format(x))
    df[f'DISTANCE_FROM_{point_name}_KM'] = df[f'DISTANCE_FROM_{point_name}_KM'].astype(float)

    # print(f'Successfully calculated distance for point: {point_name}')
    return df

def preprocess_new_offers(df_new_offers: pd.DataFrame):
    """ Preprocessing scraped offers to form which can be provided to ML model"""

    # drop accidentialy scraped wrong types of offers

    df_new_offers = df_new_offers[df_new_offers['TRANSACTION_TYPE'] == 'SELL' ]
    df_new_offers = df_new_offers[df_new_offers['ESTATE_CATEGORY'] == 'FLAT' ]
    df_new_offers = df_new_offers[df_new_offers['CITY'] == 'wroclaw' ]



    # changing name of column to clearer version
    if 'FLOOR_NO' in df_new_offers.columns:
        df_new_offers.rename(columns={ 'FLOOR_NO' : 'FLOOR_NUMBER'}, inplace=True)

    # drop not needed columns
    df_new_offers = df_new_offers.drop(columns = ['ID', 'SUBDISTRICT', 'EQUIPMENT_TYPES', 'RENT', 'STREET', 'COUNTRY',	'PROVINCE',	
                                                'SUBREGION','TRANSACTION_TYPE','ESTATE_CATEGORY',	'CITY'	]) 


    # fill missing data
    missing_categorical_cols = ['BUILDING_MATERIAL', 'WINDOWS_TYPE', 'HEATING', 'BUILDING_TYPE', 'CONSTRUCTION_STATUS', 'BUILDING_OWNERSHIP', 'DISTRICT']

    df_new_offers[missing_categorical_cols] = df_new_offers[missing_categorical_cols].fillna('unknown')
    df_new_offers['EXTRAS_TYPES'] = df_new_offers['EXTRAS_TYPES'].fillna('0')
    df_new_offers['SECURITY_TYPES'] = df_new_offers['SECURITY_TYPES'].fillna('0')


    # drop not needed columns
    df_new_offers = df_new_offers.dropna(subset = ['PRICE', 'BUILDING_FLOORS_NUM', 'FLOOR_NUMBER'])


    # correcting locations
    point_address = 'Sukiennice 14/15, 50-029 Wrocław' # (city hall address)
    point_name = 'CITY_CENTER'

    df_new_offers = add_distance_to_point(df_new_offers, point_address, point_name)

    kmeans = joblib.load('models/kmeans_locations.pkl')
    df_locations = pd.read_csv('data/locations.csv')

    # Assigning clusters to each offer in main df
    df_new_offers['CLUSTER'] = kmeans.predict(df_new_offers[['LATITUDE', 'LONGITUDE']])
    df_new_offers = pd.merge(df_new_offers,df_locations[['CLUSTER','RESIDENTIAL_AREA', 'DISTRICT']],on='CLUSTER', how='left')

    # clearing not needed columns
    df_new_offers = df_new_offers.drop(columns = ['DISTRICT_x', 'CLUSTER', 'LATITUDE', 'LONGITUDE'])
    df_new_offers = df_new_offers.rename(columns={'DISTRICT_y': 'DISTRICT'})

    df_new_offers = df_new_offers[df_new_offers['DISTANCE_FROM_CITY_CENTER_KM'] < 40 ]

    df_new_offers = create_dummies_from_features(df_new_offers, 'SECURITY_TYPES')
    df_new_offers = create_dummies_from_features(df_new_offers, 'EXTRAS_TYPES')

    categorical_columns = ['MARKET_TYPE', 'USER_TYPE','ROOMS_NUM','FLOOR_NUMBER','BUILDING_MATERIAL','BUILDING_OWNERSHIP',
                            'BUILDING_TYPE','CONSTRUCTION_STATUS','WINDOWS_TYPE','HEATING','RESIDENTIAL_AREA', 'DISTRICT']
    
    df_new_offers = add_dummies(df_new_offers, categorical_columns)

    # loading list of columns
    with open('data/columns_list.pkl', 'rb') as file:
        columns_list = pickle.load(file)

    
    # Sprawdzenie, które kolumny brakuje w dataframe
    missing_columns = set(columns_list) - set(df_new_offers.columns)

    # Dodanie brakujących kolumn z wartościami równymi 0
    for col in missing_columns:
        df_new_offers[col] = 0

    df_new_offers = df_new_offers.reindex(columns=columns_list).fillna(0)


    return df_new_offers

def fill_build_years(df: pd.DataFrame):
    """Using Random Forest Regressor model to receive missing build years.
        Due to uncertain predictions from before 2010, they will be removed in order to minimize data contamination """
    
    # extract rows with missing build years
    df_missing = df[df['BUILD_YEAR'] == 0]
    df = df[df['BUILD_YEAR'] != 0]

    print(f'Number of missing year of build: {df_missing.shape[0]}')

    # fill missing values if needed
    if df_missing.shape[0] > 0:
        # load model 
        with open('models/rf_build_year_prediction_final.pkl', 'rb') as f:
            model_rf_build_year = joblib.load(f)

        # fill values
        df_missing['BUILD_YEAR'] = model_rf_build_year.predict(df_missing.drop(columns=['BUILD_YEAR', 'SLUG']))
        df_missing['BUILD_YEAR'] = df_missing['BUILD_YEAR'].astype(int)

        # drop uncertain predictions
        df_missing = df_missing[df_missing['BUILD_YEAR'] <= 2010]
        print(f'Number of received year of build: {df_missing.shape[0]}')
        # merge dataframes
        df = pd.concat([df, df_missing], ignore_index=True)

    return df

def predict_prices_per_m2(df: pd.DataFrame):
    """Using Random Forest Regressor model to predict price per m2 for each offer """
    
    print(f'Number of loaded offers: {df.shape[0]}')

    # loading model
    with open('models/rf_price_per_m2_prediction.pkl', 'rb') as f:
        model_rf_price_per_m2 = joblib.load(f)
    
    df['PREDICTED_PRICE_PER_M2'] = model_rf_price_per_m2.predict(df.drop(columns=['SLUG', 'PRICE', 'PRICE_PER_M2']))
    

    return df