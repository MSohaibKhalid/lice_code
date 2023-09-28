#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:


import requests
import numpy as np
import pandas as pd
import datetime
from datetime import datetime
import dateutil.relativedelta as relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import json
import random
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import MinMaxScaler
from math import radians
from ast import literal_eval as convert_it
import re

from math import radians, sin, cos, sqrt, asin
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import matplotlib.pyplot as plt
import ray


# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


# cd /content/drive/MyDrive/AvgFemaleLice_Project


# In[ ]:


def get_headers_for_request(client_id, client_secret):
    ### Setting up API request
    url = "https://id.barentswatch.no/connect/token"

    payload='client_id={}&client_secret={}&grant_type=client_credentials&scope=api'.format(client_id, client_secret)
    headers = {
    'Content': 'application/x-www-form-urlencoded',
    'Content-Type': 'application/x-www-form-urlencoded'
    }

    response_token = requests.request("POST", url, headers=headers, data=payload)
    get_token = "Bearer " + response_token.text[17:-56]

    headers = {"Authorization" : get_token}

    return headers


# In[ ]:


def write_All_Localities_Treatment_Data(client_id = None, client_secret = None, locality_list = None):

    assert (client_id != None or client_secret != None or locality_list != None), "Please enter Client ID and Client Secret."

    headers = None

    df = pd.DataFrame()
    current_year = pd.Timestamp.now().year
    output_csv_path = 'final_treatment.csv'

    for loop_num, loc_num in enumerate(locality_list):
        if (loop_num % 200) == 0:
            headers = get_headers_for_request(client_id, client_secret)

        try:
            url = "https://www.barentswatch.no/bwapi/v1/geodata/fishhealth/locality/{}/liceMedicationEvents/{}".format(loc_num, current_year)
            response_treat = requests.get(url, headers = headers).json()
            weeks_data = response_treat["data"]

            listYears = []
            ListWeeks = []
            listMechanicalTreatment = []
            listMechanicalEntirity = []
            listChemicalTreatment = []
            listChemicalEntirity = []

            # Loop through each week's data
            for week in weeks_data:
                week_num = week["week"]
                cleanerFishTreatments = week.get("cleanerFishTreatments", [])
                medicinalTreatments = week.get("medicinalTreatments", [])

                listMechanicalTreatment.append(week["mechanicalRemoval"])
                listMechanicalEntirity.append(week["mechanicalRemovalEntireLocality"])

                # Loop through each treatment in the week's data
                chemicalTreatment = (len(cleanerFishTreatments + medicinalTreatments) > 0)
                chemicalEntirity = False

                for treatment in cleanerFishTreatments + medicinalTreatments:
                    chemicalEntirity = (chemicalEntirity or treatment.get("entireLocality", False))

                ListWeeks.append(week_num)
                listYears.append(current_year)
                listChemicalTreatment.append(chemicalTreatment)
                listChemicalEntirity.append(chemicalEntirity)

            df = pd.DataFrame({
                "localityNo": [loc_num]*len(ListWeeks),
                "year": listYears,
                "week": ListWeeks,
                "mechanicalTreatment": listMechanicalTreatment,
                "mechanicalEntirity": listMechanicalEntirity,
                "chemicalTreatment": listChemicalTreatment,
                "chemicalEntirity": listChemicalEntirity,
            })

            if len(df) > 0:
                df.to_csv(output_csv_path, mode='a', header=False, index=False)

        except:
            print("--> Error fetching data for Locality Number: {} for year: {}\n".format(loc_num, current_year))

    df = pd.read_csv(output_csv_path, header = None)
    df.columns = ["localityNo", "year", "week", "mechanicalTreatment", "mechanicalEntirity", "chemicalTreatment", "chemicalEntirity"]
    df = df.drop_duplicates(subset=['localityNo', 'year', 'week'], keep="last")
    df.to_csv(output_csv_path, index=False)


# In[ ]:


def write_All_Localities_Temperature_Data(client_id = None, client_secret = None, locality_list = None):

    assert (client_id != None or client_secret != None or locality_list != None), "Please enter Client ID and Client Secret."

    headers = None

    df = pd.DataFrame()
    current_year = pd.Timestamp.now().year
    output_csv_path = 'final_temperature.csv'

    for loop_num, loc_num in enumerate(locality_list):
        if (loop_num % 200) == 0:
            headers = get_headers_for_request(client_id, client_secret)

        try:
            url = "https://www.barentswatch.no/bwapi/v1/geodata/fishhealth/locality/{}/seatemperature/{}".format(loc_num, current_year)
            response_temp = requests.get(url, headers = headers).json()

            week_data = response_temp["data"]
            week_temperatures = [week["seaTemperature"] if week["seaTemperature"] is not None else 0.0 for week in week_data]

            # Create a DataFrame for this file's data
            df = pd.DataFrame({
                "localityNo": [loc_num] * len(week_temperatures),
                "year": [current_year] * len(week_temperatures),
                "week": [week["week"] for week in week_data],
                "temperature": week_temperatures
            })

            if len(df) > 0:
                df.to_csv(output_csv_path, mode='a', header=False, index=False)

        except:
            print("--> Error fetching data for Locality Number: {} for year: {}\n".format(loc_num, current_year))

    df = pd.read_csv(output_csv_path, header = None)
    df.columns = ["localityNo", "year", "week", "temperature"]
    df = df.drop_duplicates(subset=['localityNo', 'year', 'week'], keep="last")
    df.to_csv(output_csv_path, index=False)


# In[ ]:


def write_All_Localities_avgFL_Data(client_id = None, client_secret = None, locality_list = None):

    assert (client_id != None or client_secret != None or locality_list != None), "Please enter Client ID and Client Secret."

    headers = None

    df = pd.DataFrame()
    current_year = pd.Timestamp.now().year
    output_csv_path = 'final_avgFL_new.csv'

    for loop_num, loc_num in enumerate(locality_list):
        if (loop_num % 200) == 0:
            headers = get_headers_for_request(client_id, client_secret)

        try:
            url = "https://www.barentswatch.no/bwapi/v1/geodata/fishhealth/locality/{}/avgfemalelice/{}".format(loc_num, current_year)
            response_avgfemalelice = requests.get(url, headers=headers).json()

            week_data = response_avgfemalelice["data"]
            week_values = [week["value"] if week["value"] is not None else 0.0 for week in week_data]

            # Create a DataFrame for this file's data
            df = pd.DataFrame({
                "localityNo": [loc_num] * len(week_values),
                "year": [current_year] * len(week_values),
                "week": [week["week"] for week in week_data],
                "value": week_values
            })

            if len(df) > 0:
                df.to_csv(output_csv_path, mode='a', header=False, index=False)

        except:
            print("--> Error fetching data for Locality Number: {} for year: {}\n".format(loc_num, current_year))

    df = pd.read_csv(output_csv_path, header = None)
    df.columns = ["localityNo", "year", "week", "value"]
    df = df.drop_duplicates(subset=['localityNo', 'year', 'week'], keep="last")
    df.to_csv(output_csv_path, index=False)


# In[ ]:


def write_All_Localities_LiceType_Data(client_id = None, client_secret = None, locality_list = None):

    assert (client_id != None or client_secret != None or locality_list != None), "Please enter Client ID and Client Secret."

    headers = None

    df = pd.DataFrame()
    current_year = pd.Timestamp.now().year
    output_csv_path = 'final_LiceType.csv'

    for loop_num, loc_num in enumerate(locality_list):
        if (loop_num % 200) == 0:
            headers = get_headers_for_request(client_id, client_secret)

        try:
            url = "https://www.barentswatch.no/bwapi/v1/geodata/fishhealth/locality/{}/liceTypeDistribution/{}".format(loc_num, current_year)
            response_liceTypeDistribution = requests.get(url, headers=headers).json()

            x = pd.DataFrame(response_liceTypeDistribution)
            df = pd.DataFrame(x['data'].tolist())
            df["localityNo"] = [loc_num]*len(df)
            df["year"] = [current_year]*len(df)
            df[["avgMobileLice","avgStationaryLice", "week"]]
            df = df[["localityNo", "year", "week", "avgMobileLice","avgStationaryLice"]]

            if len(df) > 0:
                df.to_csv(output_csv_path, mode='a', header=False, index=False)

        except:
            print("--> Error fetching data for Locality Number: {} for year: {}\n".format(loc_num, current_year))

    df = pd.read_csv(output_csv_path, header = None)
    df.columns = ["localityNo", "year", "week", "avgMobileLice","avgStationaryLice"]
    df = df.drop_duplicates(subset=['localityNo', 'year', 'week'], keep="last")
    df.to_csv(output_csv_path, index=False)


# In[ ]:


def preprocess_data(avgfemalelice_df):
    # Find the minimum and maximum week numbers for the minimum and maximum years
    min_year = avgfemalelice_df['year'].min()
    max_year = avgfemalelice_df['year'].max()
    min_week = int(avgfemalelice_df[avgfemalelice_df['year'] == min_year]['week'].min())
    max_week = int(avgfemalelice_df[avgfemalelice_df['year'] == max_year]['week'].max())

    # Create a list of all possible week numbers for each year
    all_weeks = range(1, 53)

    # Create a new DataFrame to store the modified data
    modified_df = pd.DataFrame(columns=avgfemalelice_df.columns)

    # Iterate over each locality
    for locality in avgfemalelice_df['localityNo'].unique():
        locality_df = avgfemalelice_df[avgfemalelice_df['localityNo'] == locality]
        max_week = int(avgfemalelice_df[avgfemalelice_df['year'] == max_year]['week'].max())

        # Create a list of week numbers with zeros for missing weeks
        locality_weeks = []
        for year in range(min_year, max_year + 1):
            year_df = locality_df[locality_df['year'] == year]
            year_weeks = year_df['week'].unique()
            for week in all_weeks:
                if week in year_weeks:
                    value = year_df[year_df['week'] == week]['value'].values[0]
                    locality_weeks.append(value)
                else:
                    locality_weeks.append(0.0)

        # Create a DataFrame for the locality with the modified week numbers
        locality_modified_df = pd.DataFrame({'localityNo': [locality] * len(all_weeks) * (max_year - min_year + 1),
                                             'year': sorted(list(range(min_year, max_year + 1)) * len(all_weeks)),
                                             'week': list(all_weeks) * (max_year - min_year + 1),
                                             'value': locality_weeks})

        locality_modified_df = locality_modified_df[ : max_week-52]

        # Append the locality_modified_df to the modified_df
        modified_df = pd.concat([modified_df, locality_modified_df])

    # Sort the modified DataFrame by 'localityNo', 'year', and 'week'
    modified_df = modified_df.sort_values(by=['localityNo', 'year', 'week']).reset_index(drop=True)

    # Convert the 'value' column to numeric data type
    modified_df['value'] = pd.to_numeric(modified_df['value'], errors='coerce')

    return modified_df


# In[ ]:


def get_Localities_List(client_id = None, client_secret = None):

    assert (client_id != None or client_secret != None), "Please enter Client ID and Client Secret."

    headers = get_headers_for_request(client_id, client_secret)

    url = "https://www.barentswatch.no/bwapi/v1/geodata/fishhealth/localities"
    response_localities = requests.get(url, headers = headers).json()

    return list(pd.DataFrame(response_localities)["localityNo"])


# In[ ]:


def convert_list(str_inp):
    if isinstance(str_inp, str):
        str_inp = str_inp.strip()
        str_inp = re.sub(' +', ' ', str_inp)
        str_inp = str_inp.replace("[ ", "[")
        str_inp = str_inp.replace(" ]", "]")
        # str_inp = str_inp.replace(",,", ",")
        if "," not in str_inp:
            return convert_it(str_inp.replace(" ", ", "))
        else:
            return convert_it(str_inp)
    return str_inp


# In[ ]:


def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    distance = r * c

    return distance


def find_closest_localities(df, all_df , max_distance = 10):
    result = []

    for index, row in df.iterrows():
        localities_with_distance = []
        cur_loc = int(row['localityNo'])
        for idx, r in all_df.iterrows():
            other_loc = int(r['localityNo'])
            if cur_loc != other_loc:
                distance = haversine_distance(row['latitude'], row['longitude'], r['latitude'], r['longitude'])
                if distance <= max_distance:
                    localities_with_distance.append({'localityNo': other_loc, 'distance': distance})
        # Sort localities by distance
        localities_with_distance.sort(key=lambda x: x['distance'])
        closest_localities = [locality['localityNo'] for locality in localities_with_distance]
        result.append({'localityNo': cur_loc, 'closest_localities': closest_localities})

    result_df = pd.DataFrame(result)
    return result_df


# Define the training data generator
def generate_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


# In[ ]:


def get_Latest_Data(client_id = "msohaibkhalid96@gmail.com:bwopenapi", client_secret = "dygsjquul4pm"):
    # localities_list = get_Localities_List(client_id, client_secret)
    # write_All_Localities_avgFL_Data(client_id, client_secret, localities_list)
    # write_All_Localities_Treatment_Data(client_id, client_secret, localities_list)
    # write_All_Localities_Temperature_Data(client_id, client_secret, localities_list)
    # write_All_Localities_LiceType_Data(client_id, client_secret, localities_list)

    avgfemalelice_df = pd.read_csv('final_avgFL_new.csv')
    modified_avgFL = preprocess_data(avgfemalelice_df)

    temp_df = pd.read_csv("final_temperature.csv")

    merged_df = pd.merge(modified_avgFL, temp_df, on=['localityNo', 'year', 'week'], how='left')
    merged_df['temperature'] = merged_df['temperature'].fillna(0)

    treat_df = pd.read_csv("final_treatment.csv")
    treat_df["mechanicalTreatment"] = treat_df["mechanicalTreatment"].astype(int)
    treat_df["mechanicalEntirity"] = treat_df["mechanicalEntirity"].astype(int)
    treat_df["chemicalTreatment"] = treat_df["chemicalTreatment"].astype(int)
    treat_df["chemicalEntirity"] = treat_df["chemicalEntirity"].astype(int)

    merged_df = pd.merge(merged_df, treat_df, on=['localityNo', 'year', 'week'], how='left')

    liceType_df = pd.read_csv("final_LiceType.csv")

    merged_df = pd.merge(merged_df, liceType_df, on=['localityNo', 'year', 'week'], how='left')
    merged_df = merged_df.fillna(0)

    pos_df = pd.read_csv("position.csv")
    pos_df = pos_df.drop_duplicates(subset=['localityNo'], keep="last").reset_index(drop=True)

    merged_df = pd.merge(merged_df, pos_df, on=['localityNo'], how='left')

    merged_df.to_csv("preprocessed_data.csv", index=False)
    print('DONE')


# In[ ]:


# get_Latest_Data(client_id = "msohaibkhalid96@gmail.com:bwopenapi", client_secret = "dygsjquul4pm")


# In[ ]:


def convert_list(str_inp):
    if isinstance(str_inp, str):
        str_inp = str_inp.strip()
        str_inp = re.sub(' +', ' ', str_inp)
        str_inp = str_inp.replace("[ ", "[")
        str_inp = str_inp.replace(" ]", "]")
        # str_inp = str_inp.replace(",,", ",")
        if "," not in str_inp:
            return convert_it(str_inp.replace(" ", ", "))
        else:
            return convert_it(str_inp)
    return str_inp


# In[ ]:


# !pip install ray


# In[ ]:


# Define a Transformer model with time-based attention
def create_transformer_model(window_size, num_features):
    inputs = keras.Input(shape=(window_size, num_features))

    # Positional encoding for time-based attention
    position_embeddings = layers.Embedding(input_dim=window_size, output_dim=num_features)(tf.range(window_size))
    inputs_with_position = inputs + position_embeddings

    # Multi-head self-attention layer with more heads and key dimension
    transformer_layer = layers.MultiHeadAttention(num_heads=8, key_dim=128)(inputs_with_position, inputs_with_position)
    transformer_layer = layers.Dropout(0.2)(transformer_layer)

    # Feed-forward neural network layer (position-wise feed-forward) with more units
    transformer_layer = layers.Conv1D(256, kernel_size=1, activation='relu')(transformer_layer)
    transformer_layer = layers.Dropout(0.2)(transformer_layer)

    # Add a residual connection with projection and layer normalization
    projection = layers.Conv1D(256, kernel_size=1)(inputs)
    transformer_layer = layers.Add()([projection, transformer_layer])
    transformer_layer = layers.LayerNormalization(epsilon=1e-6)(transformer_layer)

    # Global average pooling
    transformer_layer = layers.GlobalAveragePooling1D()(transformer_layer)

    # Pass through a dense layer with more units
    outputs = layers.Dense(128, activation='relu')(transformer_layer)
    outputs = layers.Dense(1)(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_LSTM_Model(window_size, num_features):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window_size, num_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(8))
    model.add(Dense(1))
    return model

def create_BiLSTM_Model(window_size, num_features):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(window_size, num_features)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1))
    return model

# Define the training data generator for multivariate LSTM
def generate_multivariate_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

# Define the test data generator for multivariate LSTM
def get_sequences(data, window_size, pred, i):
    X = np.array(data[i:i + window_size])
    if i > 0:
        X[window_size-1, 0] = pred
    data[i:i + window_size] = X
    return np.array(X), data


def calculate_array_differences(matched_window, check_window):
    # Find the last non-zero values
    last_nonzero_matched = matched_window[matched_window.nonzero()][-1]
    last_nonzero_check = check_window[check_window.nonzero()][-1]

    # Calculate the mean using only non-zero values
    nonzero_mean_matched = np.mean(matched_window[matched_window.nonzero()])
    nonzero_mean_check = np.mean(check_window[check_window.nonzero()])

    # Calculate the differences
    diff_last_nonzero = last_nonzero_matched - last_nonzero_check
    diff_nonzero_mean = nonzero_mean_matched - nonzero_mean_check

    return diff_last_nonzero, diff_nonzero_mean


def get_decay_weights(df, col, cond):
    cond_ind = df.index[cond].tolist()
    differences = []
    tot = 0.0

    for i in cond_ind:
        true_value = df[col].iloc[i]
        next_values = df[col].iloc[i+1:i+6]
        average = next_values.mean()
        difference = np.abs(true_value - average)
        differences.append(difference)

    weigthed = list(np.array(differences)/np.sum(differences))

    return differences, weigthed


def get_weighted_sum(df, col, weights):
    values = None
    if isinstance(col, str):
        values = df[col].to_numpy()
    else:
        values = np.array(col)

    assert len(values) == len(weights), "Lengths are not same."

    weighted_sum = np.sum(np.array(weights)*np.array(values))

    return weighted_sum


def get_week_for_treatment(arr, threshold):
    arr = np.array(arr)
    indices = np.argwhere(arr >= threshold)
    if indices.size > 0:
        return indices[0][0]
    else:
        return None


def get_next_weeks(cond, n = 5):
    temp = []
    for i in range(1,n+1):
        temp.append(cond.shift(i).replace(np.nan, False))

    final_cond = None
    for i in range(n):
        if i == 0:
            final_cond = temp[i]
        else:
            final_cond = final_cond | temp[i]

    return final_cond


# In[ ]:


ray.init()

@ray.remote
def get_N_forecasts(df, given_locality = 13677, N = 5, top_k = 10, lr = 1e-3, n_epoch = 1, window_size=10, batch_size=8, output_all='all_results.csv', output_best='best_results.csv'):

    print("#"*100)
    print("Generating results for Locality Number:", given_locality)

    # try:
    given_locality_df = df[df['localityNo'] == given_locality]

    # Get the last non-zero year and week for the given locality
    last_week_data = given_locality_df[given_locality_df['value'] != 0].tail(1).reset_index(drop=True)
    year = last_week_data.year.min()
    week = last_week_data[last_week_data["year"] == year]['week'].min()

    # Filter the data for the specific week and year
    week_year_data = pd.DataFrame()
    week_year_data = pd.concat([week_year_data, given_locality_df[given_locality_df['year'] < year]])
    week_year_data = pd.concat([week_year_data, given_locality_df[(given_locality_df['week'] < week) & (given_locality_df['year'] == year)]])

    given_locality_df = week_year_data.copy()

    # Calculate the correlation coefficients for each locality
    correlation_coeffs = {}
    for locality in df['localityNo'].unique():
        if locality != given_locality:

            locality_df = df[df['localityNo'] == locality]
            # Filter the data for the specific week and year
            week_year_data = pd.DataFrame()
            week_year_data = pd.concat([week_year_data, locality_df[locality_df['year'] < year]])
            week_year_data = pd.concat([week_year_data, locality_df[(locality_df['week'] < week) & (locality_df['year'] == year)]])
            locality_df = week_year_data.copy()

            std_given_locality = given_locality_df['value'].std()
            std_locality = locality_df['value'].std()
            if std_given_locality != 0 and std_locality != 0:
                correlation = np.corrcoef(given_locality_df['value'], locality_df['value'], rowvar=False)[0, 1]
                correlation_coeffs[locality] = correlation

    # Sort the correlation coefficients in descending order and get the top K correlated localities
    top_k_correlated = sorted(correlation_coeffs.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Extract the localityNos from the top K correlated localities
    top_k_localities = [locality for locality, _ in top_k_correlated]

    print("\nTop {} most correlated localities:".format(top_k))
    for locality in top_k_localities:
        print(locality)


    ################################ Preparing DATA ################################
    # Create a DataFrame to store the training data
    data = pd.DataFrame()
    # Iterate over each week to prepare the training data
    for locality in (list(top_k_localities) + [given_locality]):
        filtered_locality_df = df[df["localityNo"] == locality]
        # Filter the data for the specific week and year
        week_year_data = pd.DataFrame()
        week_year_data = pd.concat([week_year_data, filtered_locality_df[filtered_locality_df['year'] < year]])
        week_year_data = pd.concat([week_year_data, filtered_locality_df[(filtered_locality_df['week'] < week) & (filtered_locality_df['year'] == year)]])

        if (locality == given_locality):
            data['value'] = list(week_year_data["value"])+[0]*5
            data["temperature"] = [0]*5+list(week_year_data["temperature"])
            data["mechanicalTreatment"] = [0]*5+list(week_year_data["mechanicalTreatment"])
            data["mechanicalEntirity"] = [0]*5+list(week_year_data["mechanicalEntirity"])
            data["chemicalTreatment"] = [0]*5+list(week_year_data["chemicalTreatment"])
            data["chemicalEntirity"] = [0]*5+list(week_year_data["chemicalEntirity"])
            data["avgMobileLice"] = [0]*5+list(week_year_data["avgMobileLice"])
            data["avgStationaryLice"] = [0]*5+list(week_year_data["avgStationaryLice"])
        else:
            data[str(locality)] = [0]*5+list(week_year_data["value"])
            data[str(locality) + "_temperature"] = [0]*5+list(week_year_data["temperature"])
            data[str(locality) + "_mechanicalTreatment"] = [0]*5+list(week_year_data["mechanicalTreatment"])
            data[str(locality) + "_mechanicalEntirity"] = [0]*5+list(week_year_data["mechanicalEntirity"])
            data[str(locality) + "_chemicalTreatment"] = [0]*5+list(week_year_data["chemicalTreatment"])
            data[str(locality) + "_chemicalEntirity"] = [0]*5+list(week_year_data["chemicalEntirity"])
            data[str(locality) + "_avgMobileLice"] = [0]*5+list(week_year_data["avgMobileLice"])
            data[str(locality) + "_avgStationaryLice"] = [0]*5+list(week_year_data["avgStationaryLice"])

    ################################ Preparing TRAINING DATA ################################
    # Create a DataFrame to store the training data
    training_data = data.iloc[5:-5].copy()

    y_train = training_data['value']
    training_data = training_data.drop('value', axis=1)
    # Standardize the training features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(training_data)

    ################################ Preparing TESTING DATA ################################
    # Create a DataFrame to store the training data
    testing_data = data.tail(2*N).copy()

    y_test = testing_data['value'].values
    testing_data = testing_data.drop('value', axis=1)

    # Standardize the testing features
    X_test_scaled = scaler.transform(testing_data)


    ################################ TRAINING LR MODEL ################################
    # Train a linear regression model
    modelLR = LinearRegression()
    modelLR.fit(X_train_scaled, y_train)

    print("\n--> Training LR Complete.")

    ################################ TRAINING NN MODEL ################################
    optm = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    # Train a neural network model
    modelNN = Sequential()
    modelNN.add(Dense(64, activation='relu', input_shape=(training_data.shape[1],)))
    modelNN.add(Dense(64, activation='relu'))
    modelNN.add(Dense(1))
    modelNN.compile(optimizer=optm, loss='mean_squared_error')
    modelNN.fit(X_train_scaled, y_train, epochs=n_epoch, verbose=0)

    print("\n--> Training NN Complete.")

    ################################ TRAINING NN2 MODEL ################################
    # Train a more complex neural network model
    modelNN2 = Sequential()
    modelNN2.add(Dense(128, activation='sigmoid', input_shape=(training_data.shape[1],)))
    modelNN2.add(Dense(64, activation='sigmoid'))
    modelNN2.add(Dense(8, activation='sigmoid'))
    modelNN2.add(Dense(1))
    modelNN2.compile(optimizer=optm, loss='mean_squared_error')
    modelNN2.fit(X_train_scaled, y_train, epochs=n_epoch, verbose=0)

    print("\n--> Training NN2 Complete.\n")


    ################################ TESTING LR MODEL ################################
    # Predict the values for the given locality using the model
    forecasted_valuesLR = modelLR.predict(X_test_scaled)
    forecasted_valuesLR = np.absolute(forecasted_valuesLR)
    # forecasted_valuesLR[forecasted_valuesLR < 0.0] = 0.0

    # Calculate mean absolute error and R^2 score
    maeLR = mean_absolute_error(y_test[:5], forecasted_valuesLR[:5])


    ################################ TESTING NN MODEL ################################
    # Predict the values for the given locality using the model
    forecasted_valuesNN = modelNN.predict(X_test_scaled).flatten()
    forecasted_valuesNN = np.absolute(forecasted_valuesNN)
    # forecasted_valuesNN[forecasted_valuesNN < 0.0] = 0.0

    # Calculate mean absolute error and R^2 score
    maeNN = mean_absolute_error(y_test[:5], forecasted_valuesNN[:5])

    ################################ TESTING NN MODEL ################################
    # Predict the values for the given locality using the model
    forecasted_valuesNN2 = modelNN2.predict(X_test_scaled).flatten()
    forecasted_valuesNN2 = np.absolute(forecasted_valuesNN2)
    # forecasted_valuesNN2[forecasted_valuesNN2 < 0.0] = 0.0

    # Calculate mean absolute error and R^2 score
    maeNN2 = mean_absolute_error(y_test[:5], forecasted_valuesNN2[:5])


    ##############################################################################################################
    ##############################################################################################################

    ################################ Preparing TRAINING DATA ################################
    # Create a DataFrame to store the training data
    data_sequential = data[['value', 'temperature', 'mechanicalTreatment', 'mechanicalEntirity', 'chemicalTreatment', 'chemicalEntirity', 'avgMobileLice', 'avgStationaryLice']]
    training_data = data_sequential.iloc[5:-5].copy()

    ################################ Multivariate LSTM ################################
    # Combine target variable and features for multivariate data
    multivariate_train_data = training_data.values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_train_data = scaler.fit_transform(multivariate_train_data)

    X_train, y_train = generate_multivariate_sequences(normalized_train_data, window_size)

    # Build the LSTM model with multiple input features
    model_MultiLSTM = create_LSTM_Model(window_size, multivariate_train_data.shape[1])
    model_MultiLSTM.compile(loss='mean_squared_error', optimizer='adam')
    model_MultiLSTM.fit(X_train, y_train, epochs=n_epoch, batch_size=batch_size, verbose=False)

    print("\n--> Training MultiLSTM Complete.\n")

    # Build the BiLSTM model with multiple input features
    model_MultiBiLSTM = create_BiLSTM_Model(window_size, multivariate_train_data.shape[1])
    model_MultiBiLSTM.compile(loss='mean_squared_error', optimizer='adam')
    model_MultiBiLSTM.fit(X_train, y_train, epochs=n_epoch, batch_size=batch_size, verbose=False)

    print("\n--> Training MutliBiLSTM Complete.\n")

    # Build the Transformer model with multiple input features
    model_Transformer = create_transformer_model(window_size, multivariate_train_data.shape[1])
    model_Transformer.compile(loss='mean_squared_error', optimizer='adam')
    model_Transformer.fit(X_train, y_train, epochs=n_epoch, batch_size=batch_size, verbose=False)

    print("\n--> Training Transformer Complete.\n")

    ################################ Preparing TESTING DATA ################################
    # Create a DataFrame to store the testing data
    testing_data = data_sequential.tail(2*N).copy()

    ################################ TESTING LSTM MODEL ################################
    # Prepare testing data for multivariate LSTM
    multivariate_test_data = testing_data.values
    normalized_test_data = scaler.transform(multivariate_test_data)
    y_test = multivariate_test_data[:,0].copy()
    normalized_test_data[:,:N] = np.zeros(N)

    normalized_test_data_ = np.concatenate((normalized_train_data[-(window_size+N):-N], normalized_test_data), axis=0)
    next_pred = 0.0
    forecast = []
    for i in range(2*N):
        last_window, normalized_test_data_ = get_sequences(normalized_test_data_, window_size, next_pred, i)
        next_pred = model_MultiLSTM.predict(last_window.reshape(1, window_size, multivariate_train_data.shape[1]), verbose=False)[0, 0]
        forecast.append(next_pred)
        if i == 2*N-1:
            last_window, normalized_test_data_ = get_sequences(normalized_test_data_, window_size, next_pred, i+1)
    model_MultiLSTM_preds = scaler.inverse_transform(normalized_test_data_[-2*N:])
    forecast_values_MultiLSTM = np.abs(model_MultiLSTM_preds[:,0].reshape(-1))
    # Calculate mean absolute error
    mae_MultiLSTM = mean_absolute_error(y_test[:N], forecast_values_MultiLSTM[:N])

    normalized_test_data_ = np.concatenate((normalized_train_data[-(window_size+N):-N], normalized_test_data), axis=0)
    next_pred = 0.0
    forecast = []
    for i in range(2*N):
        last_window, normalized_test_data_ = get_sequences(normalized_test_data_, window_size, next_pred, i)
        next_pred = model_MultiBiLSTM.predict(last_window.reshape(1, window_size, multivariate_train_data.shape[1]), verbose=False)[0, 0]
        forecast.append(next_pred)
        if i == 2*N-1:
            last_window, normalized_test_data_ = get_sequences(normalized_test_data_, window_size, next_pred, i+1)
    model_MultiBiLSTM_preds = scaler.inverse_transform(normalized_test_data[-2*N:])
    forecast_values_MultiBiLSTM = np.abs(model_MultiBiLSTM_preds[:,0].reshape(-1))
    # Calculate mean absolute error
    mae_MultiBiLSTM = mean_absolute_error(y_test[:N], forecast_values_MultiBiLSTM[:N])

    normalized_test_data_ = np.concatenate((normalized_train_data[-(window_size+N):-N], normalized_test_data), axis=0)
    next_pred = 0.0
    forecast = []
    for i in range(2*N):
        last_window, normalized_test_data_ = get_sequences(normalized_test_data_, window_size, next_pred, i)
        next_pred = model_Transformer.predict(last_window.reshape(1, window_size, multivariate_train_data.shape[1]), verbose=False)[0, 0]
        forecast.append(next_pred)
        if i == 2*N-1:
            last_window, normalized_test_data_ = get_sequences(normalized_test_data_, window_size, next_pred, i+1)
    model_Transformer_preds = scaler.inverse_transform(normalized_test_data[-2*N:])
    forecast_values_Transformer = np.abs(model_Transformer_preds[:,0].reshape(-1))
    # Calculate mean absolute error
    mae_Transformer = mean_absolute_error(y_test[:N], forecast_values_Transformer[:N])

    ##############################################################################################################
    ##############################################################################################################

    data_rolling = data[['value'] + [str(loc) for loc in top_k_localities]]
    training_data = data_rolling.iloc[5:-5].copy()
    trainY = training_data['value']
    trainX = training_data[[str(loc) for loc in top_k_localities]]

    testing_data = data_rolling.tail(2*N).copy()
    y_test = testing_data['value'].values.flatten()

    # Step 3: Rolling window correlation
    top_locality = None  # To store the locality with the highest correlation
    best_window = None  # To store the range of the best window
    best_correlation = -1.0  # To store the maximum correlation found so far

    check_window = pd.Series(trainY[-13:].values.flatten())  # Last 13 weeks of trainY

    # Iterate over columns in trainX
    for column in trainX.columns:
        max_correlation = -1.0  # To store the maximum correlation found for the current column
        max_window = None  # To store the range of the window with the maximum correlation

        # Iterate over each week in the column
        for i in range(len(trainX) - 12 - 2*N):
            current_window = trainX[column].values[i:i+13]  # Extract the window of 13 weeks

            # Calculate correlation between check_window and current_window
            correlation = check_window.corr(pd.Series(current_window.flatten()))

            # Update the maximum correlation and window if necessary
            if correlation > max_correlation:
                max_correlation = correlation
                max_window = (i, i+12)  # Store the start and end indices of the window

        # Update the best locality, window, and correlation if necessary
        if max_correlation > best_correlation:
            best_correlation = max_correlation
            top_locality = column
            best_window = max_window

    # Step 4: Forecasting
    prediction_last_nonzero = np.zeros(N)
    prediction_mean = np.zeros(N)
    next_5_weeks_last_nonzero = np.zeros(N)
    next_5_weeks_mean = np.zeros(N)

    if top_locality is not None and best_window is not None:
        check_window = np.array(check_window)
        matched_window = np.array(trainX[top_locality].values[best_window[0]:best_window[1]].flatten())
        forecast_window = np.array(trainX[top_locality].values[best_window[1]+1 : best_window[1]+N+1].flatten())  # Next 5 weeks after the best window
        roll_next_5_weeks = np.array(trainX[top_locality].values[best_window[1]+N+1 : best_window[1]+2*N+1].flatten())
        try:
            diff_last_nonzero, diff_nonzero_mean = calculate_array_differences(matched_window, check_window)

            prediction_last_nonzero =  np.abs(forecast_window - diff_last_nonzero)
            next_5_weeks_last_nonzero =  np.abs(roll_next_5_weeks - diff_last_nonzero)
            prediction_mean =  np.abs(forecast_window - diff_nonzero_mean)
            next_5_weeks_mean = np.abs(roll_next_5_weeks - diff_nonzero_mean)
        except:
            pass

    mae_last_nonzero = mean_absolute_error(y_test[:N], prediction_last_nonzero)
    mae_mean = mean_absolute_error(y_test[:N], prediction_mean)

    print("\n--> Training Rolling Window Complete.\n")

    ##############################################################################################################
    ##############################################################################################################
    total = maeNN + mae_mean
    perc_NN = 1 - (maeNN / total)
    perc_mean = 1 - (mae_mean / total)
    prediction_NN_mean = (perc_NN * np.array(forecasted_valuesNN[:N])) + (perc_mean * np.array(prediction_mean))
    last5weeks_NN_mean = (perc_NN * np.array(forecasted_valuesNN[N:])) + (perc_mean * np.array(next_5_weeks_mean))
    mae_NN_mean = mean_absolute_error(y_test[:N], prediction_NN_mean)

    total = maeNN + mae_last_nonzero
    perc_NN = 1 - (maeNN / total)
    perc_last_nonzero = 1 - (mae_last_nonzero / total)
    prediction_NN_last_nonzero = (perc_NN * np.array(forecasted_valuesNN[:N])) + (perc_last_nonzero * np.array(prediction_last_nonzero))
    last5weeks_NN_last_nonzero = (perc_NN * np.array(forecasted_valuesNN[N:])) + (perc_last_nonzero * np.array(next_5_weeks_last_nonzero))
    mae_NN_last_nonzero = mean_absolute_error(y_test[:N], prediction_NN_last_nonzero)


    total = maeNN2 + mae_mean
    perc_NN2 = 1 - (maeNN2 / total)
    perc_mean = 1 - (mae_mean / total)
    prediction_NN2_mean = (perc_NN2 * np.array(forecasted_valuesNN2[:N])) + (perc_mean * np.array(prediction_mean))
    last5weeks_NN2_mean = (perc_NN2 * np.array(forecasted_valuesNN2[N:])) + (perc_mean * np.array(next_5_weeks_mean))
    mae_NN2_mean = mean_absolute_error(y_test[:N], prediction_NN2_mean)

    total = maeNN2 + mae_last_nonzero
    perc_NN2 = 1 - (maeNN2 / total)
    perc_last_nonzero = 1 - (mae_last_nonzero / total)
    prediction_NN2_last_nonzero = (perc_NN2 * np.array(forecasted_valuesNN2[:N])) + (perc_last_nonzero * np.array(prediction_last_nonzero))
    last5weeks_NN2_last_nonzero = (perc_NN2 * np.array(forecasted_valuesNN2[N:])) + (perc_last_nonzero * np.array(next_5_weeks_last_nonzero))
    mae_NN2_last_nonzero = mean_absolute_error(y_test[:N], prediction_NN2_last_nonzero)



    total = mae_MultiLSTM + mae_mean
    perc_MultiLSTM = 1 - (mae_MultiLSTM / total)
    perc_mean = 1 - (mae_mean / total)
    prediction_MultiLSTM_mean = (perc_MultiLSTM * np.array(forecast_values_MultiLSTM[:N])) + (perc_mean * np.array(prediction_mean))
    last5weeks_MultiLSTM_mean = (perc_MultiLSTM * np.array(forecast_values_MultiLSTM[N:])) + (perc_mean * np.array(next_5_weeks_mean))
    mae_MultiLSTM_mean = mean_absolute_error(y_test[:N], prediction_MultiLSTM_mean)

    total = mae_MultiLSTM + mae_last_nonzero
    perc_MultiLSTM = 1 - (mae_MultiLSTM / total)
    perc_last_nonzero = 1 - (mae_last_nonzero / total)
    prediction_MultiLSTM_last_nonzero = (perc_MultiLSTM * np.array(forecast_values_MultiLSTM[:N])) + (perc_last_nonzero * np.array(prediction_last_nonzero))
    last5weeks_MultiLSTM_last_nonzero = (perc_MultiLSTM * np.array(forecast_values_MultiLSTM[N:])) + (perc_last_nonzero * np.array(next_5_weeks_last_nonzero))
    mae_MultiLSTM_last_nonzero = mean_absolute_error(y_test[:N], prediction_MultiLSTM_last_nonzero)



    total = mae_MultiBiLSTM + mae_mean
    perc_MultiBiLSTM = 1 - (mae_MultiBiLSTM / total)
    perc_mean = 1 - (mae_mean / total)
    prediction_MultiBiLSTM_mean = (perc_MultiBiLSTM * np.array(forecast_values_MultiBiLSTM[:N])) + (perc_mean * np.array(prediction_mean))
    last5weeks_MultiBiLSTM_mean = (perc_MultiBiLSTM * np.array(forecast_values_MultiBiLSTM[N:])) + (perc_mean * np.array(next_5_weeks_mean))
    mae_MultiBiLSTM_mean = mean_absolute_error(y_test[:N], prediction_MultiBiLSTM_mean)

    total = mae_MultiBiLSTM + mae_last_nonzero
    perc_MultiBiLSTM = 1 - (mae_MultiBiLSTM / total)
    perc_last_nonzero = 1 - (mae_last_nonzero / total)
    prediction_MultiBiLSTM_last_nonzero = (perc_MultiBiLSTM * np.array(forecast_values_MultiBiLSTM[:N])) + (perc_last_nonzero * np.array(prediction_last_nonzero))
    last5weeks_MultiBiLSTM_last_nonzero = (perc_MultiBiLSTM * np.array(forecast_values_MultiBiLSTM[N:])) + (perc_last_nonzero * np.array(next_5_weeks_last_nonzero))
    mae_MultiBiLSTM_last_nonzero = mean_absolute_error(y_test[:N], prediction_MultiBiLSTM_last_nonzero)


    total = mae_Transformer + mae_mean
    perc_Transformer = 1 - (mae_Transformer / total)
    perc_mean = 1 - (mae_mean / total)
    prediction_Transformer_mean = (perc_Transformer * np.array(forecast_values_Transformer[:N])) + (perc_mean * np.array(prediction_mean))
    last5weeks_Transformer_mean = (perc_Transformer * np.array(forecast_values_Transformer[N:])) + (perc_mean * np.array(next_5_weeks_mean))
    mae_Transformer_mean = mean_absolute_error(y_test[:N], prediction_Transformer_mean)

    total = mae_Transformer + mae_last_nonzero
    perc_Transformer = 1 - (mae_Transformer / total)
    perc_last_nonzero = 1 - (mae_last_nonzero / total)
    prediction_Transformer_last_nonzero = (perc_Transformer * np.array(forecast_values_Transformer[:N])) + (perc_last_nonzero * np.array(prediction_last_nonzero))
    last5weeks_Transformer_last_nonzero = (perc_Transformer * np.array(forecast_values_Transformer[N:])) + (perc_last_nonzero * np.array(next_5_weeks_last_nonzero))
    mae_Transformer_last_nonzero = mean_absolute_error(y_test[:N], prediction_Transformer_last_nonzero)

    print("\n--> Training Combined Models Complete.\n")

    # comb = list(arr1 + arr2)
    # comb_list.append(comb)


    ##############################################################################################################
        ########################################### Best Model ################################################

    all_mae = [maeLR, maeNN, maeNN2, mae_MultiLSTM, mae_MultiBiLSTM, mae_Transformer, mae_mean, mae_last_nonzero, mae_NN_mean,
    mae_NN_last_nonzero, mae_NN2_mean, mae_NN2_last_nonzero, mae_MultiLSTM_mean, mae_MultiLSTM_last_nonzero,
    mae_MultiBiLSTM_mean, mae_MultiBiLSTM_last_nonzero, mae_Transformer_mean, mae_Transformer_last_nonzero]

    all_preds = [forecasted_valuesLR[:N], forecasted_valuesNN[:N], forecasted_valuesNN2[:N], forecast_values_MultiLSTM[:N],
    forecast_values_MultiBiLSTM[:N], forecast_values_Transformer[:N], prediction_mean, prediction_last_nonzero, prediction_NN_mean,
    prediction_NN_last_nonzero, prediction_NN2_mean, prediction_NN2_last_nonzero, prediction_MultiLSTM_mean,
    prediction_MultiLSTM_last_nonzero, prediction_MultiBiLSTM_mean, prediction_MultiBiLSTM_last_nonzero,
    prediction_Transformer_mean, prediction_Transformer_last_nonzero]

    all_next5weeks = [forecasted_valuesLR[N:], forecasted_valuesNN[N:], forecasted_valuesNN2[N:], forecast_values_MultiLSTM[N:],
    forecast_values_MultiBiLSTM[N:], forecast_values_Transformer[N:], next_5_weeks_mean, next_5_weeks_last_nonzero,
    last5weeks_NN_mean, last5weeks_NN_last_nonzero, last5weeks_NN2_mean, last5weeks_NN2_last_nonzero, last5weeks_MultiLSTM_mean,
    last5weeks_MultiLSTM_last_nonzero, last5weeks_MultiBiLSTM_mean, last5weeks_MultiBiLSTM_last_nonzero,
    last5weeks_Transformer_mean, last5weeks_Transformer_last_nonzero]

    all_models = ['LR', 'NN', 'NN2', 'MultiLSTM', 'MultiBiLSTM', 'Transformer', 'rollingMean', 'rollingLastNonZero', 'NN_mean',
     'NN_last_nonzero', 'NN2_mean', 'NN2_last_nonzero', 'MultiLSTM_mean', 'MultiLSTM_last_nonzero',
     'MultiBiLSTM_mean', 'MultiBiLSTM_last_nonzero', 'Transformer_mean', 'Transformer_last_nonzero']

    best_mae = min(all_mae)
    best_mae_idx = all_mae.index(best_mae)
    best_preds = all_preds[best_mae_idx]
    best_next5weeks = all_next5weeks[best_mae_idx]
    best_model_name = all_models[best_mae_idx]


    ##############################################################################################################
        ########################################### Treatment ################################################

    loc_treatments = {}
    lice_threshold_dict = {}
    avgMobileLice_threshold_dict = {}
    avgStationaryLice_threshold_dict = {}
    lice_decay_dict = {}

    preds = best_next5weeks.copy()

    lice_threshold = 1000.0
    avgLice_decay_val = 0.0

    if True:
    # if np.any(np.array(preds) >= 0.5):
        for loc in top_k_localities+[given_locality]:
            # Filter the DataFrame for the locality with the highest count
            chunk = df[df['localityNo'] == loc].reset_index(drop=True)
            cond = ((chunk['mechanicalTreatment'] == 1) | (chunk['chemicalTreatment'] == 1)) & (chunk['value'] >= min(preds))
            cond = cond.shift(-1).replace(np.nan, False)
            decays, contributions = get_decay_weights(chunk, "value", cond)

            treat_chunk = chunk[cond]
            if len(treat_chunk) != 0:
                loc_treatments[loc] = len(treat_chunk)
                lice_threshold_dict[loc] = get_weighted_sum(treat_chunk, "value", contributions)
                avgMobileLice_threshold_dict[loc] = get_weighted_sum(treat_chunk, "avgMobileLice", contributions)
                avgStationaryLice_threshold_dict[loc] = get_weighted_sum(treat_chunk, "avgStationaryLice", contributions)
                lice_decay_dict[loc] = get_weighted_sum(treat_chunk, decays, contributions)

        weights = {}
        total_treatmnets = sum(loc_treatments.values())

        lice_threshold = 0.0
        if total_treatmnets != 0.0:
            for k, v in loc_treatments.items():
                weights[k] = v/total_treatmnets

            avgMobileLice_threshold = 0.0
            avgStationaryLice_threshold = 0.0
            avgLice_decay_val = 0.0
            for loc, w in weights.items():
                lice_threshold += w*lice_threshold_dict[loc]
                avgMobileLice_threshold += w*avgMobileLice_threshold_dict[loc]
                avgStationaryLice_threshold += w*avgStationaryLice_threshold_dict[loc]
                avgLice_decay_val += w*lice_decay_dict[loc]

        else:
            lice_threshold = 0.0

    treatment_week = get_week_for_treatment(preds, lice_threshold)
    week_for_treatment = -1
    todo_treatment = "No"
    preds_with_decay = list(preds)

    if treatment_week is not None:
        todo_treatment = "Yes"
        for i in range(treatment_week,5):
            preds_with_decay[i] = (preds_with_decay[i]-avgLice_decay_val) if (preds_with_decay[i]-avgLice_decay_val > 0.0) else 0.0
        week_for_treatment = treatment_week+1


    if lice_threshold == 1000.0:
        lice_threshold = 0.0

    lice_threshold, avgLice_decay_val, week_for_treatment, todo_treatment

    ##############################################################################################################
    ##############################################################################################################


    temp_df = pd.DataFrame({
        "localityNo": [given_locality],
        "actual_values": [str(y_test[:N])],
        "data_points": [(given_locality_df['value'] != 0).sum()],

        "LR_MAE": [maeLR],
        "LR_Preds": [str(forecasted_valuesLR[:N])],
        "LR_next5weeks": [str(forecasted_valuesLR[N:])],

        "NN_MAE": [maeNN],
        "NN_Preds": [str(forecasted_valuesNN[:N])],
        "NN_next5weeks": [str(forecasted_valuesNN[N:])],

        "NN2_MAE": [maeNN2],
        "NN2_Preds": [str(forecasted_valuesNN2[:N])],
        "NN2_next5weeks": [str(forecasted_valuesNN2[N:])],

        "MultiLSTM_MAE": [mae_MultiLSTM],
        "MultiLSTM_Preds": [str(forecast_values_MultiLSTM[:N])],
        "MultiLSTM_next5weeks": [str(forecast_values_MultiLSTM[N:])],

        "MultiBiLSTM_MAE": [mae_MultiBiLSTM],
        "MultiBiLSTM_Preds": [str(forecast_values_MultiBiLSTM[:N])],
        "MultiBiLSTM_next5weeks": [str(forecast_values_MultiBiLSTM[N:])],

        "Transformer_MAE": [mae_Transformer],
        "Transformer_Preds": [str(forecast_values_Transformer[:N])],
        "Transformer_next5weeks": [str(forecast_values_Transformer[N:])],

        "rollingMean_MAE": [mae_mean],
        "rollingMean_Preds": [str(prediction_mean)],
        "rollingMean_next5weeks": [str(next_5_weeks_mean)],

        "rollingLastNonZero_MAE": [mae_last_nonzero],
        "rollingLastNonZero_Preds": [str(prediction_last_nonzero)],
        "rollingLastNonZero_next5weeks": [str(next_5_weeks_last_nonzero)],

        "NN_rollingMean_MAE": [mae_NN_mean],
        "NN_rollingMean_Preds": [str(prediction_NN_mean)],
        "NN_rollingMean_next5weeks": [str(last5weeks_NN_mean)],
        "NN_rollingMean_MAE": [mae_NN_last_nonzero],
        "NN_rollingMean_Preds": [str(prediction_NN_last_nonzero)],
        "NN_rollingMean_next5weeks": [str(last5weeks_NN_last_nonzero)],

        "NN2_rollingMean_MAE": [mae_NN2_mean],
        "NN2_rollingMean_Preds": [str(prediction_NN2_mean)],
        "NN2_rollingMean_next5weeks": [str(last5weeks_NN2_mean)],
        "NN2_rollingMean_MAE": [mae_NN2_last_nonzero],
        "NN2_rollingMean_Preds": [str(prediction_NN2_last_nonzero)],
        "NN2_rollingMean_next5weeks": [str(last5weeks_NN2_last_nonzero)],

        "MultiLSTM_rollingMean_MAE": [mae_MultiLSTM_mean],
        "MultiLSTM_rollingMean_Preds": [str(prediction_MultiLSTM_mean)],
        "MultiLSTM_rollingMean_next5weeks": [str(last5weeks_MultiLSTM_mean)],
        "MultiLSTM_rollingMean_MAE": [mae_MultiLSTM_last_nonzero],
        "MultiLSTM_rollingMean_Preds": [str(prediction_MultiLSTM_last_nonzero)],
        "MultiLSTM_rollingMean_next5weeks": [str(last5weeks_MultiLSTM_last_nonzero)],

        "MultiBiLSTM_rollingMean_MAE": [mae_MultiBiLSTM_mean],
        "MultiBiLSTM_rollingMean_Preds": [str(prediction_MultiBiLSTM_mean)],
        "MultiBiLSTM_rollingMean_next5weeks": [str(last5weeks_MultiBiLSTM_mean)],
        "MultiBiLSTM_rollingMean_MAE": [mae_MultiBiLSTM_last_nonzero],
        "MultiBiLSTM_rollingMean_Preds": [str(prediction_MultiBiLSTM_last_nonzero)],
        "MultiBiLSTM_rollingMean_next5weeks": [str(last5weeks_MultiBiLSTM_last_nonzero)],

        "Transformer_rollingMean_MAE": [mae_Transformer_mean],
        "Transformer_rollingMean_Preds": [str(prediction_Transformer_mean)],
        "Transformer_rollingMean_next5weeks": [str(last5weeks_Transformer_mean)],
        "Transformer_rollingMean_MAE": [mae_Transformer_last_nonzero],
        "Transformer_rollingMean_Preds": [str(prediction_Transformer_last_nonzero)],
        "Transformer_rollingMean_next5weeks": [str(last5weeks_Transformer_last_nonzero)],
    })

    temp_df.to_csv(output_all, mode='a', header=False, index=False)

    ##############################################################################################################
    ##############################################################################################################

    best_df = pd.DataFrame({
        "localityNo": [given_locality],
        "data_points": [(given_locality_df['value'] != 0).sum()],
        "actual_values": [str(y_test[:N])],
        "best_model": [best_model_name],
        "mae": [best_mae],
        "preds": [str(best_next5weeks)],
        "todo_treatment": [todo_treatment],
        "week_for_treatment": [week_for_treatment],
        "treatment_threshold": [lice_threshold],
        "expected_decay": [avgLice_decay_val],
        "preds_with_decay": [str(preds_with_decay)],
    })

    best_df.to_csv(output_best, mode='a', header=False, index=False)

    ##############################################################################################################
    ##############################################################################################################


    return given_locality

    # except:
    #     print("\n--> Error processing data for Locality Number: {}\n".format(given_locality))


# In[ ]:

if __name__=="__main__":

    # get_N_forecasts(df)
    import boto3
    import pandas as pd

    # Configure AWS credentials. Replace 'YOUR_ACCESS_KEY' and 'YOUR_SECRET_KEY' with your own credentials.
    aws_access_key = 'AKIARRDJRCPCINZURGHA'
    aws_secret_key = '6TKsd+kWfq59ID3iserxmpiGnP/lNrMDKoRj9J1C'
    # aws_session_token = 'YOUR_SESSION_TOKEN'  # Optional, if you're using temporary session-based credentials

    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

    bucket_name = 'mylice'
    file_key = 'preprocessed_data.csv'

    # Specify a local file name where you want to save the downloaded CSV file.
    local_file_name = 'preprocessed_data.csv'

    # Download the file from S3 to your local machine.
    s3.download_file(bucket_name, file_key, local_file_name)

    # Read the CSV file into a DataFrame.
    df = pd.read_csv(local_file_name) 

    # localities_list = get_Localities_List(client_id = "msohaibkhalid96@gmail.com:bwopenapi", client_secret = "dygsjquul4pm")
    localities_list = df['localityNo'].unique().tolist()


    # In[ ]:
    output_all_file_name='all_results.csv'
    output_best_file_name='best_results.csv'
    futures = [get_N_forecasts.remote(df=df, given_locality = loc, output_all = output_all_file_name, output_best = output_best_file_name) for loc in localities_list]

    # Upload the local CSV file to S3.
    s3.upload_file(output_all_file_name, bucket_name, output_all_file_name)
    s3.upload_file(output_best_file_name, bucket_name, output_best_file_name)
    

    # In[ ]:


    print(ray.get(futures))


    # In[ ]:


    # for loc in localities_list:
    #     get_N_forecasts.remote(df=df, given_locality = loc)


    # In[ ]:


    # temp_df_cols = ['localityNo', 'actual_values', 'data_points', 'LR_MAE', 'LR_Preds', 'LR_next5weeks', 'NN_MAE', 'NN_Preds', 'NN_next5weeks', 'NN2_MAE', 'NN2_Preds', 'NN2_next5weeks', 'MultiLSTM_MAE', 'MultiLSTM_Preds', 'MultiLSTM_next5weeks', 'MultiBiLSTM_MAE', 'MultiBiLSTM_Preds', 'MultiBiLSTM_next5weeks', 'Transformer_MAE', 'Transformer_Preds', 'Transformer_next5weeks', 'rollingMean_MAE', 'rollingMean_Preds', 'rollingMean_next5weeks', 'rollingLastNonZero_MAE', 'rollingLastNonZero_Preds', 'rollingLastNonZero_next5weeks', 'NN_rollingMean_MAE', 'NN_rollingMean_Preds', 'NN_rollingMean_next5weeks', 'NN2_rollingMean_MAE', 'NN2_rollingMean_Preds', 'NN2_rollingMean_next5weeks', 'MultiLSTM_rollingMean_MAE', 'MultiLSTM_rollingMean_Preds', 'MultiLSTM_rollingMean_next5weeks', 'MultiBiLSTM_rollingMean_MAE', 'MultiBiLSTM_rollingMean_Preds', 'MultiBiLSTM_rollingMean_next5weeks', 'Transformer_rollingMean_MAE', 'Transformer_rollingMean_Preds', 'Transformer_rollingMean_next5weeks']

    # How to read best_results.csv
    best_df = pd.read_csv('./All_Localities/best_results.csv', header = None)
    best_df_cols = ['localityNo', 'data_points', 'actual_values', 'best_model', 'mae', 'preds', 'todo_treatment', 'week_for_treatment', 'treatment_threshold', 'expected_decay', 'preds_with_decay']
    best_df.columns = best_df_cols
    best_df.head()


    # In[ ]:




