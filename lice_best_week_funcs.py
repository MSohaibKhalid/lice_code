import requests
import numpy as np
import pandas as pd
import datetime
import os
from math import radians
from ast import literal_eval as convert_it
import re
from math import radians, sin, cos, sqrt, asin
import traceback
import ray

from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow import keras
from tensorflow.keras import layers

import logging

logging.getLogger("cmdstanpy").disabled = True # Turns 'cmdstanpy' logs off


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



def write_All_Localities_Treatment_Data(output_csv_path, client_id = None, client_secret = None, locality_list = None):

    assert (client_id != None or client_secret != None or locality_list != None), "Please enter Client ID and Client Secret."

    headers = None

    df = pd.DataFrame()
    current_year = pd.Timestamp.now().year

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

    df = pd.read_csv(output_csv_path)
    df = df.drop_duplicates(subset=['localityNo', 'year', 'week'], keep="last")
    df.to_csv(output_csv_path, index=False)



def write_All_Localities_Temperature_Data(output_csv_path, client_id = None, client_secret = None, locality_list = None):

    assert (client_id != None or client_secret != None or locality_list != None), "Please enter Client ID and Client Secret."

    headers = None

    df = pd.DataFrame()
    current_year = pd.Timestamp.now().year

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

    df = pd.read_csv(output_csv_path)
    df = df.drop_duplicates(subset=['localityNo', 'year', 'week'], keep="last")
    df.to_csv(output_csv_path, index=False)



def write_All_Localities_avgFL_Data(output_csv_path, client_id = None, client_secret = None, locality_list = None):

    assert (client_id != None or client_secret != None or locality_list != None), "Please enter Client ID and Client Secret."

    headers = None

    df = pd.DataFrame()
    current_year = pd.Timestamp.now().year

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

    df = pd.read_csv(output_csv_path)
    df = df.drop_duplicates(subset=['localityNo', 'year', 'week'], keep="last")
    df.to_csv(output_csv_path, index=False)




def write_All_Localities_LiceType_Data(output_csv_path, client_id = None, client_secret = None, locality_list = None):

    assert (client_id != None or client_secret != None or locality_list != None), "Please enter Client ID and Client Secret."

    headers = None

    df = pd.DataFrame()
    current_year = pd.Timestamp.now().year

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

    df = pd.read_csv(output_csv_path)
    df = df.drop_duplicates(subset=['localityNo', 'year', 'week'], keep="last")
    df.to_csv(output_csv_path, index=False)



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



def get_Localities_List(client_id = None, client_secret = None):
    headers = get_headers_for_request(client_id, client_secret)

    url = "https://www.barentswatch.no/bwapi/v1/geodata/fishhealth/localities"
    response_localities = requests.get(url, headers = headers).json()

    return list(pd.DataFrame(response_localities)["localityNo"])



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



def get_Latest_Data(avgFL_file, temperature_file, treatment_file, liceType_file, output_file,
                    client_id = None, client_secret = None, fetch_new_data = False):
    
    assert (client_id != None or client_secret != None), "Please enter Client ID and Client Secret."

    if fetch_new_data:
        localities_list = get_Localities_List(client_id, client_secret)
        write_All_Localities_avgFL_Data(avgFL_file, client_id, client_secret, localities_list)
        write_All_Localities_Temperature_Data(temperature_file, client_id, client_secret, localities_list)
        write_All_Localities_Treatment_Data(treatment_file, client_id, client_secret, localities_list)
        write_All_Localities_LiceType_Data(liceType_file, client_id, client_secret, localities_list)

    avgfemalelice_df = pd.read_csv(avgFL_file)
    modified_avgFL = preprocess_data(avgfemalelice_df)

    temp_df = pd.read_csv(temperature_file)

    merged_df = pd.merge(modified_avgFL, temp_df, on=['localityNo', 'year', 'week'], how='left')
    merged_df['temperature'] = merged_df['temperature'].fillna(0)

    treat_df = pd.read_csv(treatment_file)
    treat_df["mechanicalTreatment"] = treat_df["mechanicalTreatment"].astype(int)
    treat_df["mechanicalEntirity"] = treat_df["mechanicalEntirity"].astype(int)
    treat_df["chemicalTreatment"] = treat_df["chemicalTreatment"].astype(int)
    treat_df["chemicalEntirity"] = treat_df["chemicalEntirity"].astype(int)

    merged_df = pd.merge(merged_df, treat_df, on=['localityNo', 'year', 'week'], how='left')

    liceType_df = pd.read_csv(liceType_file)

    merged_df = pd.merge(merged_df, liceType_df, on=['localityNo', 'year', 'week'], how='left')
    merged_df = merged_df.fillna(0)

    # pos_df = pd.read_csv("position.csv")
    # pos_df = pos_df.drop_duplicates(subset=['localityNo'], keep="last").reset_index(drop=True)

    # merged_df = pd.merge(merged_df, pos_df, on=['localityNo'], how='left')

    merged_df.to_csv(output_file, index=False)
    print('DONE')



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



def get_top_K_localities(given_locality, given_locality_df, non_zero_entries_indices, df, top_k):
    # Calculate the correlation coefficients for each locality
    correlation_coeffs = {}
    for locality in df['localityNo'].unique():
        if locality != given_locality:
            locality_df = df[df['localityNo'] == locality].reset_index(drop=True).copy()
            locality_df = locality_df[non_zero_entries_indices].reset_index(drop=True)
            
            # # Filter the data for the specific week and year
            # week_year_data = pd.DataFrame()
            # week_year_data = pd.concat([week_year_data, locality_df[locality_df['year'] < year]])
            # week_year_data = pd.concat([week_year_data, locality_df[(locality_df['week'] <= week) & (locality_df['year'] == year)]])
            # locality_df = week_year_data.copy()

            std_given_locality = given_locality_df['value'].std()
            std_locality = locality_df['value'].std()
            if std_given_locality != 0 and std_locality != 0:
                correlation = np.corrcoef(given_locality_df['value'], locality_df['value'], rowvar=False)[0, 1]
                correlation_coeffs[locality] = correlation

    # Sort the correlation coefficients in descending order and get the top K correlated localities
    top_k_correlated = sorted(correlation_coeffs.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Extract the localityNos from the top K correlated localities
    top_k_localities = [locality for locality, _ in top_k_correlated]

    return top_k_localities


def extend_values(df, col, n):
    df_loc = df.copy()

    df_loc['ds'] = pd.to_datetime(df_loc['year'].astype(str) + df_loc['week'].astype(str) + '0', format='%Y%U%w')
    df_loc.rename(columns={col: 'y'}, inplace=True)
    df_loc = df_loc[['ds', 'y']].reset_index(drop=True)

    # Create and fit the Prophet model
    model = Prophet(weekly_seasonality=True)
    model.fit(df_loc)
    # Create a dataframe with the future dates for forecasting
    future = model.make_future_dataframe(periods=n, freq='W')  # Forecasting for the next 5 days

    # Generate the forecast
    forecast = model.predict(future)
    return list(forecast['yhat'])



def prepare_dataset(df, given_locality, non_zero_entries_indices, top_k_localities, year, week, n, run):
    # Create a DataFrame to store the training data
    data = pd.DataFrame()
    temp_df = df[df['localityNo'] == given_locality].reset_index(drop=True)
    indices = temp_df[non_zero_entries_indices].index
    actuals = temp_df[non_zero_entries_indices]['value'].values[-2*n:]

    if run == "first":
        indices = indices[:-2*n]
        actuals = actuals[:n]
    else:
        indices = indices[:-n]
        actuals = actuals[n:]
    
    # Iterate over each week to prepare the training data
    for locality in (list(top_k_localities) + [given_locality]):
        locality_df = df[df['localityNo'] == locality].reset_index(drop=True)
        locality_df = locality_df.loc[indices].reset_index(drop=True)
        
        if (locality == given_locality):
            data['value'] = list(locality_df["value"])+list(actuals)
            data["temperature"] = extend_values(locality_df, 'temperature', n)
            data["mechanicalTreatment"] = list(locality_df["mechanicalTreatment"])+[0]*n
            data["mechanicalEntirity"] = list(locality_df["mechanicalEntirity"])+[0]*n
            data["chemicalTreatment"] = list(locality_df["chemicalTreatment"])+[0]*n
            data["chemicalEntirity"] = list(locality_df["chemicalEntirity"])+[0]*n
            data["avgMobileLice"] = extend_values(locality_df, 'avgMobileLice', n)
            data["avgStationaryLice"] = extend_values(locality_df, 'avgStationaryLice', n)
        else:
            data[str(locality)] = extend_values(locality_df, 'value', n)
            data[str(locality) + "_temperature"] = extend_values(locality_df, 'temperature', n)
            data[str(locality) + "_mechanicalTreatment"] = list(locality_df["mechanicalTreatment"])+[0]*n
            data[str(locality) + "_mechanicalEntirity"] = list(locality_df["mechanicalEntirity"])+[0]*n
            data[str(locality) + "_chemicalTreatment"] = list(locality_df["chemicalTreatment"])+[0]*n
            data[str(locality) + "_chemicalEntirity"] = list(locality_df["chemicalEntirity"])+[0]*n
            data[str(locality) + "_avgMobileLice"] = extend_values(locality_df, 'avgMobileLice', n)
            data[str(locality) + "_avgStationaryLice"] = extend_values(locality_df, 'avgStationaryLice', n)

    return data


def get_results_LR(model, X_train_scaled, y_train, ts, X_test_sample, actual_val, best_model_specs, N, run):
    
    model_name = "LR"

    if run == 'first':
        # Train a linear regression model
        if model is None:
            model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        print("\n--> Training {} Complete.".format(model_name))

        # Predict the values for the given locality using the model
        pred = model.predict(X_test_sample)
        pred = np.absolute(pred[0])

        mae = np.absolute(pred - actual_val)
        
        print("--> Evaluation for {} Complete.".format(model_name))

        if mae < best_model_specs['weeks_test_mae'][ts]:
            best_model_specs['weeks_test_model_name'][ts] = model_name
            best_model_specs['weeks_test_preds'][ts] = pred
            best_model_specs['weeks_test_mae'][ts] = mae
            print('Updating best model specs for {}'.format(model_name))
        print()
    

    elif run == "second" and best_model_specs["weeks_test_model_name"][ts] == model_name:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_sample)
        pred = np.absolute(pred[0])
        mae = np.absolute(pred - actual_val)
        best_model_specs['weeks_future_values'][ts] = pred

    else:
        model, mae, pred, best_model_specs = model, None, None, best_model_specs

    return model, mae, pred, best_model_specs



def get_results_NN(model, X_train_scaled, y_train, ts, X_test_sample, actual_val, best_model_specs, n_epoch, lr, N, run):
    
    model_name = "NN"
    optm = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    if run == 'first':
        # Train a neural network model
        if model is None:
            model = Sequential()
            model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer=optm, loss='mean_absolute_error')
        model.fit(X_train_scaled, y_train, epochs=n_epoch, verbose=0)

        print("\n--> Training {} Complete.".format(model_name))

        # Predict the values for the given locality using the model
        pred = model.predict(X_test_sample)
        pred = np.absolute(pred[0])

        # Calculate mean absolute error and R^2 score
        mae = np.absolute(pred - actual_val)

        print("--> Evaluation for {} Complete.".format(model_name))

        if mae < best_model_specs['weeks_test_mae'][ts]:
            best_model_specs['weeks_test_model_name'][ts] = model_name
            best_model_specs['weeks_test_preds'][ts] = pred
            best_model_specs['weeks_test_mae'][ts] = mae
            print('Updating best model specs for {}'.format(model_name))
        print()

    elif run == "second" and best_model_specs["weeks_test_model_name"][ts] == model_name:
        model.fit(X_train_scaled, y_train, epochs=n_epoch, verbose=0)
        pred = model.predict(X_test_sample)
        pred = np.absolute(pred[0])
        mae = np.absolute(pred - actual_val)
        best_model_specs['weeks_future_values'][ts] = pred

    else:
        model, mae, pred, best_model_specs = model, None, None, best_model_specs

    return model, mae, pred, best_model_specs



def get_results_NN2(model, X_train_scaled, y_train, ts, X_test_sample, actual_val, best_model_specs, n_epoch, lr, N, run):
    
    model_name = "NN2"
    optm = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    if run == 'first':
        # Train a more complex neural network model
        if model is None:
            model = Sequential()
            model.add(Dense(128, activation='sigmoid', input_shape=(X_train_scaled.shape[1],)))
            model.add(Dense(64, activation='sigmoid'))
            model.add(Dense(8, activation='sigmoid'))
            model.add(Dense(1))
            model.compile(optimizer=optm, loss='mean_absolute_error')
        model.fit(X_train_scaled, y_train, epochs=n_epoch, verbose=0)

        print("\n--> Training {} Complete.".format(model_name))

        # Predict the values for the given locality using the model
        pred = model.predict(X_test_sample)
        pred = np.absolute(pred[0])

        # Calculate mean absolute error and R^2 score
        mae = np.absolute(pred - actual_val)

        print("--> Evaluation for {} Complete.".format(model_name))

        if mae < best_model_specs['weeks_test_mae'][ts]:
            best_model_specs['weeks_test_model_name'][ts] = model_name
            best_model_specs['weeks_test_preds'][ts] = pred
            best_model_specs['weeks_test_mae'][ts] = mae
            print('Updating best model specs for {}'.format(model_name))
        print()

    elif run == "second" and best_model_specs["weeks_test_model_name"][ts] == model_name:
        model.fit(X_train_scaled, y_train, epochs=n_epoch, verbose=0)
        pred = model.predict(X_test_sample)
        pred = np.absolute(pred[0])
        mae = np.absolute(pred - actual_val)
        best_model_specs['weeks_future_values'][ts] = pred

    else:
        model, mae, pred, best_model_specs = model, None, None, best_model_specs

    return model, mae, pred, best_model_specs


def get_results_MultiLSTM(model, X_train, y_train, scaled_data_chunk, X_test_sample, actual_val, scaler_seq, window_size, batch_size, best_model_specs, n_epoch, lr, N, ts, run):
    
    model_name = "MultiLSTM"
    feature_length = scaled_data_chunk.shape[1]
    optm = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    if run == 'first':
        if model is None:
            # Build the LSTM model with multiple input features
            model = create_LSTM_Model(window_size, feature_length)
            model.compile(optimizer=optm, loss='mean_absolute_error')
        model.fit(X_train, y_train, epochs=n_epoch, batch_size=batch_size, verbose=False)

        print("\n--> Training {} Complete.".format(model_name))

        normalized_test_data_ = np.concatenate((scaled_data_chunk, X_test_sample), axis=0)
        last_window = normalized_test_data_[ts:ts + window_size].copy()
        pred = model.predict(last_window.reshape(1, window_size, feature_length), verbose=False)[0, 0]
        pred = np.absolute(pred)

        # Calculate mean absolute error
        mae = np.absolute(pred - actual_val)

        print("--> Evaluation for {} Complete.".format(model_name))
    
        if mae < best_model_specs['weeks_test_mae'][ts]:
            best_model_specs['weeks_test_model_name'][ts] = model_name
            best_model_specs['weeks_test_preds'][ts] = pred
            best_model_specs['weeks_test_mae'][ts] = mae
            print('Updating best model specs for {}'.format(model_name))
        print()

    elif run == "second" and best_model_specs["weeks_test_model_name"][ts] == model_name:
        model.fit(X_train, y_train, epochs=n_epoch, batch_size=batch_size, verbose=False)
        normalized_test_data_ = np.concatenate((scaled_data_chunk, X_test_sample), axis=0)
        last_window = normalized_test_data_[ts:ts + window_size].copy()
        pred = model.predict(last_window.reshape(1, window_size, feature_length), verbose=False)[0, 0]
        pred = np.absolute(pred)
        mae = np.absolute(pred - actual_val)
        best_model_specs['weeks_future_values'][ts] = pred

    else:
        model, mae, pred, best_model_specs = model, None, None, best_model_specs

    return model, mae, pred, best_model_specs


def get_results_MultiBiLSTM(model, X_train, y_train, scaled_data_chunk, X_test_sample, actual_val, scaler_seq, window_size, batch_size, best_model_specs, n_epoch, lr, N, ts, run):
    
    model_name = "MultiBiLSTM"
    feature_length = scaled_data_chunk.shape[1]
    optm = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    
    if run == 'first':
        if model is None:
            # Build the LSTM model with multiple input features
            model = create_BiLSTM_Model(window_size, feature_length)
            model.compile(optimizer=optm, loss='mean_absolute_error')
        model.fit(X_train, y_train, epochs=n_epoch, batch_size=batch_size, verbose=False)

        print("\n--> Training {} Complete.".format(model_name))

        normalized_test_data_ = np.concatenate((scaled_data_chunk, X_test_sample), axis=0)
        last_window = normalized_test_data_[ts:ts + window_size].copy()
        pred = model.predict(last_window.reshape(1, window_size, feature_length), verbose=False)[0, 0]
        pred = np.absolute(pred)

        # Calculate mean absolute error
        mae = np.absolute(pred - actual_val)

        print("--> Evaluation for {} Complete.".format(model_name))
    
        if mae < best_model_specs['weeks_test_mae'][ts]:
            best_model_specs['weeks_test_model_name'][ts] = model_name
            best_model_specs['weeks_test_preds'][ts] = pred
            best_model_specs['weeks_test_mae'][ts] = mae

            print('Updating best model specs for {}'.format(model_name))

        print()
    
    elif run == "second" and best_model_specs["weeks_test_model_name"][ts] == model_name:
        model.fit(X_train, y_train, epochs=n_epoch, batch_size=batch_size, verbose=False)
        normalized_test_data_ = np.concatenate((scaled_data_chunk, X_test_sample), axis=0)
        last_window = normalized_test_data_[ts:ts + window_size].copy()
        pred = model.predict(last_window.reshape(1, window_size, feature_length), verbose=False)[0, 0]
        pred = np.absolute(pred)
        mae = np.absolute(pred - actual_val)
        best_model_specs['weeks_future_values'][ts] = pred

    else:
        model, mae, pred, best_model_specs = model, None, None, best_model_specs

    return model, mae, pred, best_model_specs


def get_results_transformer(model, X_train, y_train, scaled_data_chunk, X_test_sample, actual_val, scaler_seq, window_size, batch_size, best_model_specs, n_epoch, lr, N, ts, run):
    
    model_name = "transformer"
    feature_length = scaled_data_chunk.shape[1]
    optm = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    if run == 'first':
        if model is None:
            # Build the LSTM model with multiple input features
            model = create_transformer_model(window_size, feature_length)
            model.compile(optimizer=optm, loss='mean_absolute_error')
        model.fit(X_train, y_train, epochs=n_epoch, batch_size=batch_size, verbose=False)

        print("\n--> Training {} Complete.".format(model_name))

        normalized_test_data_ = np.concatenate((scaled_data_chunk, X_test_sample), axis=0)
        last_window = normalized_test_data_[ts:ts + window_size].copy()
        pred = model.predict(last_window.reshape(1, window_size, feature_length), verbose=False)[0, 0]
        pred = np.absolute(pred)

        # Calculate mean absolute error
        mae = np.absolute(pred - actual_val)

        print("--> Evaluation for {} Complete.".format(model_name))

        if mae < best_model_specs['weeks_test_mae'][ts]:
            best_model_specs['weeks_test_model_name'][ts] = model_name
            best_model_specs['weeks_test_preds'][ts] = pred
            best_model_specs['weeks_test_mae'][ts] = mae
            print('Updating best model specs for {}'.format(model_name))
        print()

    elif run == "second" and best_model_specs["weeks_test_model_name"][ts] == model_name:
        model.fit(X_train, y_train, epochs=n_epoch, batch_size=batch_size, verbose=False)
        normalized_test_data_ = np.concatenate((scaled_data_chunk, X_test_sample), axis=0)
        last_window = normalized_test_data_[ts:ts + window_size].copy()
        pred = model.predict(last_window.reshape(1, window_size, feature_length), verbose=False)[0, 0]
        pred = np.absolute(pred)
        mae = np.absolute(pred - actual_val)
        best_model_specs['weeks_future_values'][ts] = pred

    else:
        model, mae, pred, best_model_specs = model, None, None, best_model_specs

    return model, mae, pred, best_model_specs


def get_results_rolling_window(trainX, trainY, actual_values, best_model_specs, N):
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

    prediction_last_nonzero = np.nan_to_num(prediction_last_nonzero, nan=0.0)
    prediction_mean = np.nan_to_num(prediction_mean, nan=0.0)

    print("--> Evaluation for rolling_window Complete.")

    for ts in range(N):
        mae_last_nonzero = np.absolute(prediction_last_nonzero[ts] - actual_values[ts])
        mae_mean = np.absolute(prediction_mean[ts] - actual_values[ts])
        if mae_mean < best_model_specs['weeks_test_mae'][ts]:
            best_model_specs['weeks_test_model_name'][ts] = "rolling_mean"
            best_model_specs['weeks_test_preds'][ts] = prediction_mean[ts]
            best_model_specs['weeks_future_values'][ts] = next_5_weeks_mean[ts]
            best_model_specs['weeks_test_mae'][ts] = mae_mean

            print(f'Updating best model specs for rolling_mean for week {ts}')
        
        if mae_last_nonzero < best_model_specs['weeks_test_mae'][ts]:
            best_model_specs['weeks_test_model_name'][ts] = "rolling_last_nonzero"
            best_model_specs['weeks_test_preds'][ts] = prediction_last_nonzero[ts]
            best_model_specs['weeks_future_values'][ts] = next_5_weeks_last_nonzero[ts]
            best_model_specs['weeks_test_mae'][ts] = mae_last_nonzero

            print(f'Updating best model specs for rolling_last_nonzero for week {ts}')

    print()

    return mae_mean, prediction_mean.reshape(-1), next_5_weeks_mean.reshape(-1), mae_last_nonzero, prediction_last_nonzero.reshape(-1), next_5_weeks_last_nonzero.reshape(-1), best_model_specs


def get_results_comb(model_name, mae_rolling, preds_rolling, futures_rolling, mae2, pred2, actual_val, best_model_specs, ts, run):
    mae1 = mae_rolling[ts]
    total = mae1 + mae2
    if (total < 0.0001) or (np.isnan(total)):
        total = 0.0001

    pred1 = preds_rolling[ts] if run == "first" else futures_rolling[ts]

    perc1 = 1 - (mae1 / total)
    perc2 = 1 - (mae2 / total)
    pred = perc1 * pred1 + perc2 * pred2
    pred = 0.0 if np.isnan(pred) else pred

    # Calculate mean absolute error
    mae = np.absolute(pred - actual_val)

    print("--> Evaluation for {} Complete.".format(model_name))

    if run == 'first':
        if mae < best_model_specs['mae']:
            best_model_specs['weeks_test_model_name'][ts] = model_name
            best_model_specs['weeks_test_preds'][ts] = pred
            best_model_specs['weeks_test_mae'][ts] = mae

            print('Updating best model specs for {}'.format(model_name))

        print()
    
    elif run == "second" and best_model_specs["weeks_test_model_name"][ts] == model_name:
        best_model_specs['weeks_future_values'][ts] = pred

    else:
        mae, pred, best_model_specs = None, None, best_model_specs

    return mae, pred, best_model_specs


def max_difference_between_arrays(arr1, arr2):
    # Ensure both arrays are 1D
    array1 = np.ravel(arr1)
    array2 = np.ravel(arr2)
    # Calculate the absolute difference between corresponding elements
    difference_array = np.abs(array1 - array2)
    # Find the maximum difference
    max_difference = np.max(difference_array)
    return max_difference


def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res


@ray.remote
def get_N_forecasts(df, given_locality = 19015, N = 5, top_k = 10, lr = 1e-3, n_epoch = 200, window_size=10, batch_size=8, output_all='all_results.csv', output_best='best_results.csv', training_history = 'training_history.csv'):

    print("#"*100)
    print("Generating results for Locality Number:", given_locality)
    
    # Get the current date and time when the code starts
    start_time = datetime.datetime.now()

    try:
        given_locality_df = df[df['localityNo'] == given_locality].reset_index(drop=True)
        non_zero_entries_indices = (given_locality_df['value'] != 0)
        given_locality_df = given_locality_df[non_zero_entries_indices].reset_index(drop=True)

        # Get the last non-zero year and week for the given locality
        last_week_data = given_locality_df.tail(1).reset_index(drop=True)
        year = last_week_data['year'].values[0]
        week = last_week_data['week'].values[0]
        latest_lice_value = last_week_data['value'].values[0]

        # Extract the localityNos from the top K correlated localities
        top_k_localities = get_top_K_localities(given_locality, given_locality_df, non_zero_entries_indices, df, top_k)

        best_model_specs = {}
        best_model_specs['weeks_test_model_name'] = ['']*5
        best_model_specs['weeks_test_preds'] = np.zeros(N)
        best_model_specs['weeks_test_mae'] = np.ones(N)*100
        best_model_specs['weeks_future_values'] = np.zeros(N)

        model_LR = None
        model_NN = None
        model_NN2 = None
        model_MultiLSTM = None
        model_MultiBiLSTM = None
        model_transformer = None

        preds_LR = np.zeros(N)
        preds_NN = np.zeros(N)
        preds_NN2 = np.zeros(N)
        preds_MultiLSTM = np.zeros(N)
        preds_MultiBiLSTM = np.zeros(N)
        preds_transformer = np.zeros(N)
        preds_lastNonZero_NN = np.zeros(N)
        preds_lastNonZero_NN2 = np.zeros(N)
        preds_lastNonZero_MultiLSTM = np.zeros(N)
        preds_lastNonZero_MultiBiLSTM = np.zeros(N)
        preds_lastNonZero_transformer = np.zeros(N)
        preds_mean_NN = np.zeros(N)
        preds_mean_NN2 = np.zeros(N)
        preds_mean_MultiLSTM = np.zeros(N)
        preds_mean_MultiBiLSTM = np.zeros(N)
        preds_mean_transformer = np.zeros(N)

        preds_mean = np.zeros(N)
        preds_lastNonZero = np.zeros(N)
        future_mean = np.zeros(N)
        future_lastNonZero = np.zeros(N)

        best_futures = np.zeros(N)
        actual_values = np.zeros(N)

        all_results_dict = {
            "localityNo": [given_locality],
            "data_points": [non_zero_entries_indices.sum()]
        }

        for run in ["first", "second"]:
            # Create a DataFrame to store the all data for given locality
            data = prepare_dataset(df, given_locality, non_zero_entries_indices, top_k_localities, year, week, N, run)

            ################################ Preparing TRAINING DATA ################################
            # Create a DataFrame to store the training data
            training_data = data.iloc[:-N].copy()

            y_train = training_data['value']
            training_data = training_data.drop('value', axis=1)
            # Standardize the training features
            scaler_M2 = StandardScaler()
            X_train_scaled_M2 = scaler_M2.fit_transform(training_data)

            ################################ Preparing TESTING DATA ################################
            # Create a DataFrame to store the training data
            testing_data = data.tail(N).copy()

            actual_values = testing_data['value'].values
            actual_values = actual_values.reshape(-1)
            testing_data = testing_data.drop('value', axis=1)

            # Standardize the testing features
            X_test_scaled_M2 = scaler_M2.transform(testing_data)
            test_sample = None

            ################################ Preparing Sequential TRAINING DATA ################################
            # Create a DataFrame to store the training data
            data_sequential = data[['value', 'temperature', 'mechanicalTreatment', 'mechanicalEntirity', 'chemicalTreatment', 'chemicalEntirity', 'avgMobileLice', 'avgStationaryLice']]
        
            training_data_seq = data_sequential.iloc[:-N].copy()
            multivariate_train_data_seq = training_data_seq.values

            scaler_seq = StandardScaler()
            X_train_scaled_M4 = scaler_seq.fit_transform(multivariate_train_data_seq)
            X_trained_scaled_chunk = X_train_scaled_M4[-(window_size+N):-N]

            X_train_seq, y_train_seq = generate_multivariate_sequences(X_train_scaled_M4, window_size)


            ################################ Preparing Sequential TESTING DATA ################################
            # Create a DataFrame to store the testing data
            testing_data_seq = data_sequential.tail(N).copy()

            multivariate_test_data_seq = testing_data_seq.values
            X_test_scaled_M4 = scaler_seq.transform(multivariate_test_data_seq)
            X_test_scaled_M4[:,-1] = 0

            ################################ Preparing Rolling Window TRAINING DATA ################################
            if run == "first":
                data = prepare_dataset(df, given_locality, non_zero_entries_indices, top_k_localities, year, week, N, "second")
                data_rolling = data[['value'] + [str(loc) for loc in top_k_localities]]
                training_data_rolling = data_rolling.iloc[:-2*N].copy()
                y_train_rolling = training_data_rolling['value']
                X_train_rolling = training_data_rolling[[str(loc) for loc in top_k_localities]]

                mae_mean, preds_mean, future_mean, mae_lastNonZero, preds_lastNonZero, future_lastNonZero, best_model_specs = get_results_rolling_window(X_train_rolling, y_train_rolling, actual_values, best_model_specs, N)

            ##############################################################################################################
            ##############################################################################################################

            for ts in range(N):
                test_sample = X_test_scaled_M2
                if ts != 0:
                    X_test_scaled_M4[ts,-1] = best_model_specs['weeks_test_pred'][ts-1]
                test_sample_seq = X_test_scaled_M4[:ts+1]
                actual_val = actual_values[ts]

                ################################ Getting Results for M2 MODELs ################################
                model_LR, mae_LR, pred_LR, best_model_specs = get_results_LR(model_LR, X_train_scaled_M2, y_train, ts, test_sample, actual_val, best_model_specs, N, run)
                model_NN, mae_NN, pred_NN, best_model_specs = get_results_NN(model_NN, X_train_scaled_M2, y_train, ts, test_sample, actual_val, best_model_specs, n_epoch, lr, N, run)
                model_NN2, mae_NN2, pred_NN2, best_model_specs = get_results_NN2(model_NN2, X_train_scaled_M2, y_train, ts, test_sample, actual_val, best_model_specs, n_epoch, lr, N, run)


                ##############################################################################################################
                ##############################################################################################################

                ################################ Getting Results for M4 MODELs ################################
                model_MultiLSTM, mae_MultiLSTM, pred_MultiLSTM, best_model_specs = get_results_MultiLSTM(model_MultiLSTM, X_train_seq, y_train_seq, X_trained_scaled_chunk, test_sample_seq, actual_val, scaler_seq, window_size, batch_size, best_model_specs, n_epoch, lr, N, ts, run)
                model_MultiBiLSTM, mae_MultiBiLSTM, pred_MultiBiLSTM, best_model_specs = get_results_MultiBiLSTM(model_MultiBiLSTM, X_train_seq, y_train_seq, X_trained_scaled_chunk, test_sample_seq, actual_val, scaler_seq, window_size, batch_size, best_model_specs, n_epoch, lr, N, ts, run)
                model_transformer, mae_transformer, pred_transformer, best_model_specs = get_results_transformer(model_transformer, X_train_seq, y_train_seq, X_trained_scaled_chunk, test_sample_seq, actual_val, scaler_seq, window_size, batch_size, best_model_specs, n_epoch, lr, N, ts, run)
                

                ##############################################################################################################
                ##############################################################################################################
            
            
                mae_mean_NN, pred_mean_NN, best_model_specs = get_results_comb("mean_NN", mae_mean, preds_mean, future_mean, mae_NN, pred_NN, actual_val, best_model_specs, ts, run)
                mae_lastNonZero_NN, pred_lastNonZero_NN, best_model_specs = get_results_comb("lastNonZero_NN", mae_lastNonZero, preds_lastNonZero, future_lastNonZero, mae_NN, pred_NN, actual_val, best_model_specs, ts, run)

                mae_mean_NN2, pred_mean_NN2, best_model_specs = get_results_comb("mean_NN2", mae_mean, preds_mean, future_mean, mae_NN2, pred_NN2, actual_val, best_model_specs, ts, run)
                mae_lastNonZero_NN2, pred_lastNonZero_NN2, best_model_specs = get_results_comb("lastNonZero_NN2", mae_lastNonZero, preds_lastNonZero, future_lastNonZero, mae_NN2, pred_NN2, actual_val, best_model_specs, ts, run)

                mae_mean_MultiLSTM, pred_mean_MultiLSTM, best_model_specs = get_results_comb("mean_MultiLSTM", mae_mean, preds_mean, future_mean, mae_MultiLSTM, pred_MultiLSTM, actual_val, best_model_specs, ts, run)
                mae_lastNonZero_MultiLSTM, pred_lastNonZero_MultiLSTM, best_model_specs = get_results_comb("lastNonZero_MultiLSTM", mae_lastNonZero, preds_lastNonZero, future_lastNonZero, mae_MultiLSTM, pred_MultiLSTM, actual_val, best_model_specs, ts, run)

                mae_mean_MultiBiLSTM, pred_mean_MultiBiLSTM, best_model_specs = get_results_comb("mean_MultiBiLSTM", mae_mean, preds_mean, future_mean, mae_MultiBiLSTM, pred_MultiBiLSTM, actual_val, best_model_specs, ts, run)
                mae_lastNonZero_MultiBiLSTM, pred_lastNonZero_MultiBiLSTM, best_model_specs = get_results_comb("lastNonZero_MultiBiLSTM", mae_lastNonZero, preds_lastNonZero, future_lastNonZero, mae_MultiBiLSTM, pred_MultiBiLSTM, actual_val, best_model_specs, ts, run)

                mae_mean_transformer, pred_mean_transformer, best_model_specs = get_results_comb("mean_transformer", mae_mean, preds_mean, future_mean, mae_transformer, pred_transformer, actual_val, best_model_specs, ts, run)
                mae_lastNonZero_transformer, pred_lastNonZero_transformer, best_model_specs = get_results_comb("lastNonZero_transformer", mae_lastNonZero, preds_lastNonZero, future_lastNonZero, mae_transformer, pred_transformer, actual_val, best_model_specs, ts, run)

                print("\n--> Training Combined Models Complete.\n")

                ##############################################################################################################
                ##############################################################################################################

                if run == "first":
                    preds_LR[ts] = pred_LR
                    preds_NN[ts] = pred_NN
                    preds_NN2[ts] = pred_NN2
                    preds_MultiLSTM[ts] = pred_MultiLSTM
                    preds_MultiBiLSTM[ts] = pred_MultiBiLSTM
                    preds_transformer[ts] = pred_transformer
                    preds_lastNonZero_NN[ts] = pred_lastNonZero_NN
                    preds_lastNonZero_NN2[ts] = pred_lastNonZero_NN2
                    preds_lastNonZero_MultiLSTM[ts] = pred_lastNonZero_MultiLSTM
                    preds_lastNonZero_MultiBiLSTM[ts] = pred_lastNonZero_MultiBiLSTM
                    preds_lastNonZero_transformer[ts] = pred_lastNonZero_transformer
                    preds_mean_NN[ts] = pred_mean_NN
                    preds_mean_NN2[ts] = pred_mean_NN2
                    preds_mean_MultiLSTM[ts] = pred_mean_MultiLSTM
                    preds_mean_MultiBiLSTM[ts] = pred_mean_MultiBiLSTM
                    preds_mean_transformer[ts] = pred_mean_transformer

            if run == "first":
                run_dict = {
                    f"actual_values": [str(actual_values)],

                    f"LR_MAE": [mae_LR],
                    f"LR_Preds": [str(preds_LR)],

                    f"NN_MAE": [mae_NN],
                    f"NN_Preds": [str(preds_NN)],

                    f"NN2_MAE": [mae_NN2],
                    f"NN2_Preds": [str(preds_NN2)],

                    f"MultiLSTM_MAE": [mae_MultiLSTM],
                    f"MultiLSTM_Preds": [str(preds_MultiLSTM)],

                    f"MultiBiLSTM_MAE": [mae_MultiBiLSTM],
                    f"MultiBiLSTM_Preds": [str(preds_MultiBiLSTM)],

                    f"Transformer_MAE": [mae_transformer],
                    f"Transformer_Preds": [str(preds_transformer)],

                    f"rollingMean_MAE": [mae_mean],
                    f"rollingMean_Preds": [str(preds_mean)],
                    f"rollingMean_next5weeks": [str(future_mean)],

                    f"rollingLastNonZero_MAE": [mae_lastNonZero],
                    f"rollingLastNonZero_Preds": [str(preds_lastNonZero)],
                    f"rollingLastNonZero_next5weeks": [str(future_lastNonZero)],

                    f"NN_rollingMean_MAE": [mae_mean_NN],
                    f"NN_rollingMean_Preds": [str(preds_mean_NN)],
                    f"NN_rollingLastNonZero_MAE": [mae_lastNonZero_NN],
                    f"NN_rollingLastNonZero_Preds": [str(preds_lastNonZero_NN)],

                    f"NN2_rollingMean_MAE": [mae_mean_NN2],
                    f"NN2_rollingMean_Preds": [str(preds_mean_NN2)],
                    f"NN2_rollingLastNonZero_MAE": [mae_lastNonZero_NN2],
                    f"NN2_rollingLastNonZero_Preds": [str(preds_lastNonZero_NN2)],

                    f"MultiLSTM_rollingMean_MAE": [mae_mean_MultiLSTM],
                    f"MultiLSTM_rollingMean_Preds": [str(preds_mean_MultiLSTM)],
                    f"MultiLSTM_rollingLastNonZero_MAE": [mae_lastNonZero_MultiLSTM],
                    f"MultiLSTM_rollingLastNonZero_Preds": [str(preds_lastNonZero_MultiLSTM)],

                    f"MultiBiLSTM_rollingMean_MAE": [mae_mean_MultiBiLSTM],
                    f"MultiBiLSTM_rollingMean_Preds": [str(preds_mean_MultiBiLSTM)],
                    f"MultiBiLSTM_rollingLastNonZero_MAE": [mae_lastNonZero_MultiBiLSTM],
                    f"MultiBiLSTM_rollingLastNonZero_Preds": [str(preds_lastNonZero_MultiBiLSTM)],

                    f"Transformer_rollingMean_MAE": [mae_mean_transformer],
                    f"Transformer_rollingMean_Preds": [str(preds_mean_transformer)],
                    f"Transformer_rollingLastNonZero_MAE": [mae_lastNonZero_transformer],
                    f"Transformer_rollingLastNonZero_Preds": [str(preds_lastNonZero_transformer)],
                }
                
                all_results_dict = merge_dict(all_results_dict, run_dict)
        
        all_results_df = pd.DataFrame(all_results_dict)
        if os.path.isfile(output_all):
            all_results_df.to_csv(output_all, mode='a', header=False, index=False)
        else:
            all_results_df.to_csv(output_all, index=False)

        ##############################################################################################################
        ##############################################################################################################        

        best_df = pd.DataFrame({
            "localityNo": [given_locality],
            "data_points": [non_zero_entries_indices.sum()],
            "actual_values": [str(actual_values)],
            "reported_week": [week],
            "reported_year": [year],
            "latest_lice_value": [latest_lice_value],
            "neighbours": [str(top_k_localities)],
            "best_model": [str(best_model_specs['weeks_test_model_name'])],
            "mae": [str(best_model_specs['weeks_test_mae'])],
            "preds": [str(best_model_specs['weeks_test_preds'])],
            "future_values": [str(best_model_specs['week_future_values'])],
        })

        if os.path.isfile(output_best):
            best_df.to_csv(output_best, mode='a', header=False, index=False)
        else:
            best_df.to_csv(output_best, index=False)

        ##############################################################################################################
        ##############################################################################################################

        end_time = datetime.datetime.now()

        # Calculate the duration of code execution
        duration = end_time - start_time
        minutes = duration.seconds // 60
        seconds = duration.seconds % 60

        train_hist_df = pd.DataFrame({
            "localityNo": [given_locality],
            "trained": ['yes'],
            'error_message': ['N/A'],
            'traceback': ['N/A'],
            'start_time': [str(start_time)],
            'end_time': [str(end_time)],
            'duration': [str(minutes)+' mins and '+str(seconds)+' secs'],
        })
        
        if os.path.isfile(training_history):
            train_hist_df.to_csv(training_history, mode='a', header=False, index=False)
        else:
            train_hist_df.to_csv(training_history, index=False)


    
    except Exception as e:
        # Get the traceback as a string
        traceback_str = traceback.format_exc()

        end_time = datetime.datetime.now()

        # Calculate the duration of code execution
        duration = end_time - start_time
        minutes = duration.seconds // 60
        seconds = duration.seconds % 60

        train_hist_df = pd.DataFrame({
            "localityNo": [given_locality],
            "trained": ['no'],
            'error_message': [str(e)],
            'traceback': [traceback_str],
            'start_time': [str(start_time)],
            'end_time': [str(end_time)],
            'duration': [str(minutes)+' mins and '+str(seconds)+' secs'],
        })
        
        if os.path.isfile(training_history):
            train_hist_df.to_csv(training_history, mode='a', header=False, index=False)
        else:
            train_hist_df.to_csv(training_history, index=False)

        print("\n--> Error processing data for Locality Number: {}\n".format(given_locality))

    return given_locality