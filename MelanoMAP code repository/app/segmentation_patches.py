import os
import logging
import time
import pandas as pd
from tqdm import tqdm

# Our pipeline
from ai_functionalities.predict_masks import ai_segment
from handcraft_detection.handcraft_am import hc_am_detection
from handcraft_detection.handcraft_lo import hc_lo_detection

def setup_logging():
    logging.basicConfig(filename='Validation_data_error_log.txt', level=logging.ERROR)

def determine_staining(wsi_name):
    if 'am' in wsi_name.lower():
        return 'am'
    elif 'lo' in wsi_name.lower():
        return 'lo'
    else:
        raise NotImplementedError(f"For slide {wsi_name} -- Staining not implemented")

# Dataframe to store the results
def load_or_create_dataframe(results_df_name):
    if os.path.isfile(results_df_name):
        return pd.read_csv(results_df_name)
    return pd.DataFrame(columns=['File Name', 'Staining', 'HC Prediction', 'HC P-value', 'AI Prediction', 'AI P-value', 'Processing Time (s)'])

def compute_hc_results(staining, image_path):
    if staining=='am':
        return hc_am_detection(image_path)
    elif staining=='lo':
        return hc_lo_detection(image_path)

# Update file in which we store the result as we obtain them
def update_dataframe(df, wsi_name, staining, hc_overall_prediction, hc_p_value, overall_prediction, p_value, total_compute_time):
    new_row = {'File Name': wsi_name, 'Staining': staining, 'HC Prediction': hc_overall_prediction, 'HC P-value': hc_p_value, 'AI Prediction': overall_prediction, 'AI P-value': p_value, 'Processing Time (s)': total_compute_time}
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

def process_slide(wsi_name):
    image_dir = '/app/results_pipeline/patches'
    staining = determine_staining(wsi_name)
    image_path = os.path.join(image_dir, wsi_name.split('.')[0])
    ai_segment(image_path, staining=staining)

def Mel_pipeline(wsi_name):
    try:
        process_slide(wsi_name)
    except Exception as e:
        print(f"An error occurred with file {wsi_name}: {e}")
        logging.error("An error occurred with file %s: %s", wsi_name, e)


if __name__ == "__main__":
    setup_logging()
    data_dir = "/data/histos2segment"
    for wsi_name in tqdm(os.listdir(data_dir), desc='Processing whole slide images'):
       Mel_pipeline(wsi_name)

