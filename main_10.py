from lice_funcs_10 import *

import boto3
import pandas as pd
import argparse
import datetime

import ray
ray.init()


def get_arguments():
    parser = argparse.ArgumentParser(description="Lice Forecasting.")
    parser.add_argument("aws_access_key", type=str, help="AWS Access_Key")
    parser.add_argument("aws_secret_key", type=str, help="AWS Secret Key")
    parser.add_argument("aws_region_name", type=str, help="AWS Region Name")
    parser.add_argument("aws_bucket_name", type=str, help="AWS Bucket Name")
    parser.add_argument("client_id", type=str, help="API Client ID")
    parser.add_argument("client_secret", type=str, help="API Client Secret")
    parser.add_argument("fetch_new_data", type=bool, default=False, help="Download and import new data in the experiment")
    parser.add_argument("batch_size", type=int, default=25, help="Batch_size")
    parser.add_argument("max_localities", type=int, default=1500, help="Maximum number of localities to run code")
    parser.add_argument("n_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("top_k", type=int, default=10, help="Number of top correlated localities to consider")
    return parser


if __name__=="__main__":

    output_all_file_name = 'all_results_extended_data.csv'
    output_best_file_name ='best_results_extended_data.csv'
    training_history_file = 'training_history_extended_data.csv'

    info_file_name = "locality_info.csv"
    limits_file_name = "lice_limits.csv"

    avgFL_file_name = "final_avgFL.csv"
    temperature_file_name = "final_temperature.csv"
    treatment_file_name = "final_treatment.csv"
    liceType_file_name = "final_LiceType.csv"

    data_file_name = 'preprocessed_data.csv'

    # Getting all arguments necessary for forecasting
    args_parser = get_arguments()
    args = args_parser.parse_args()

    # Training parameters
    fetch_new_data = args.fetch_new_data
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    max_localities = args.max_localities
    top_k_corr = args.top_k

    # Configure API Client credentials.
    api_client_id = args.client_id
    api_client_secret = args.client_secret
    
    # Configure AWS credentials.
    aws_access_key = args.aws_access_key
    aws_secret_key = args.aws_secret_key
    aws_region_name = args.aws_region_name
    bucket_name = args.aws_bucket_name

    # Creating pointer to AWS Storage bucket
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region_name)

    # Download the file from S3 to your local machine.
    s3.download_file(bucket_name, info_file_name, info_file_name)
    s3.download_file(bucket_name, limits_file_name, limits_file_name)
    s3.download_file(bucket_name, avgFL_file_name, avgFL_file_name)
    s3.download_file(bucket_name, temperature_file_name, temperature_file_name)
    s3.download_file(bucket_name, treatment_file_name, treatment_file_name)
    s3.download_file(bucket_name, liceType_file_name, liceType_file_name)

    get_Latest_Data(avgFL_file_name, temperature_file_name, \
                    treatment_file_name, liceType_file_name, data_file_name,
                    client_id = api_client_id, client_secret = api_client_secret, \
                    fetch_new_data = fetch_new_data)
    
    s3.upload_file(data_file_name, bucket_name, data_file_name)

    # Read the CSV file into a DataFrame.
    df = pd.read_csv(data_file_name)

    try:
        s3.delete_object(Bucket=bucket_name, Key=output_all_file_name)
    except:
        print(" * Unable to delete All results file from S3 * ")

    try:
        s3.delete_object(Bucket=bucket_name, Key=output_best_file_name)
    except:
        print(" * Unable to delete Best results file from S3 * ")

    try:
        s3.delete_object(Bucket=bucket_name, Key=training_history_file)
    except:
        print(" * Unable to delete Training History file from S3 * ")


    # localities_list = df['localityNo'].unique().tolist()[:max_localities]
    localities_list = [22775, 20075, 29576, 11864, 35477, 38957, 23816, 11966, 20316, 11318, 35237, 12662, 35617, 10660, 12108, 12884, 38577, 25855, 11964, 10870, 32297, 45029, 11435, 13996, 25235, 11225, 11861, 11087, 13887, 35777, 10837, 12714, 13570, 13254, 33157, 10811, 13227, 45017, 40377, 10332, 26775, 11355, 36099, 10635, 40357, 11332, 13541, 31117, 30437, 13139, 22335, 14679, 12890, 18657, 27996, 12244, 32637, 13249, 36137, 10505, 36118, 34357, 13567, 10317, 18717, 12897, 12904, 13677, 19015, 33697, 29697, 35417, 34037]

    for i in range(0, len(localities_list), batch_size):
        batch = localities_list[i:i + batch_size]
        
        futures = [get_N_forecasts.remote(df=df, given_locality = loc, top_k = top_k_corr, n_epoch = n_epochs, \
                                           output_all = output_all_file_name, output_best = output_best_file_name, \
                                           training_history = training_history_file) \
                                            for loc in batch]
        ray.get(futures)
        
        try:
            s3.upload_file(output_all_file_name, bucket_name, output_all_file_name)
        except:
            print("No All results file to upload")
        try:
            s3.upload_file(output_best_file_name, bucket_name, output_best_file_name)
        except:
            print("No Best results file to upload")
        try:
            s3.upload_file(training_history_file, bucket_name, training_history_file)
        except:
            print("Error uploading training history file")


    best_df = pd.read_csv(output_best_file_name)
    best_df = best_df.drop_duplicates(subset=['localityNo'], keep="last").reset_index(drop=True)

    limits = pd.read_csv(limits_file_name)
    info = pd.read_csv(info_file_name)

    lice_over_limit = []
    pred_5th_over_limit = []

    for locality in best_df.localityNo.tolist():
        row_best_df = best_df[best_df['localityNo'] == locality].reset_index(drop=True)
        latest_lice_value = row_best_df['latest_lice_value'].values[0]
        pred_5th_value = row_best_df['pred_5th_value'].values[0]
        reported_week = row_best_df['reported_week'].values[0]

        row_info = info[info['localityNo'] == locality].reset_index(drop=True)
        prod_area = row_info['productionAreaID'].values[0]

        limit = limits.loc[limits['Week'] == reported_week, "PA"+str(prod_area)].values[0]

        if latest_lice_value > limit:
            lice_over_limit.append('yes')
        else:
            lice_over_limit.append('no')
        
        if pred_5th_value > limit:
            pred_5th_over_limit.append('yes')
        else:
            pred_5th_over_limit.append('no')
        
    best_df['lice_over_limit'] = lice_over_limit
    best_df['5th_pred_over_limit'] = pred_5th_over_limit
    best_df['rank'] = best_df['mae'].rank(ascending=True).astype(int)
    best_df.to_csv(output_best_file_name, index=False)
    
    s3.upload_file(output_best_file_name, bucket_name, \
                   output_best_file_name[:-4]+'_'+datetime.datetime.now().strftime('%Y-%m-%d')+'.csv')

    # s3.upload_file(avgFL_file_name, bucket_name, avgFL_file_name)
    # s3.upload_file(temperature_file_name, bucket_name, temperature_file_name)
    # s3.upload_file(treatment_file_name, bucket_name, treatment_file_name)
    # s3.upload_file(liceType_file_name, bucket_name, liceType_file_name)
    
    print('\n\n################ Results Generated ################')