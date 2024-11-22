import os
import requests
import librosa
import numpy as np
import argparse
import json
from huggingface_hub import login
from datasets import Dataset,load_dataset,concatenate_datasets,DatasetDict, Audio
from itertools import islice
import asyncio
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset From Datacreation tool")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config

def fetch_data_from_datacreation_tool(generate_specific_new_datas,start_end_value):
    # Fetch data from the API
    if generate_specific_new_datas:
        try:
            if start_end_value:
                api_url = f"https://datacreationtool.procit.com/nodeapi/getData?starting={int(start_end_value[0])}&end={int(start_end_value[1])}"
            else:
                raise ValueError("start_end_value is not defined or empty")
        except (IndexError, ValueError) as e:
            print("Error:", e)

    else:
        #api_url = "https://datacreationtool.procit.com/nodeapi/getData"
        api_url = "https://psa-dev.kcmsecurity.eu/stcai-datasetcreationapi/getData"
    headers = {
        'X-Security-Authkey': 'c14475f1-4968-4418-bb23-f242cf8070c5',  
        'Content-Type': 'application/json'
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        api_data = response.json()
    else:
        print(f"API request failed with status code {response.status_code}")
        api_data = None

    if api_data:
        data_dict = {}
        for record in api_data[0:10]:
            for key, value in record.items():
                if key not in data_dict:
                    data_dict[key] = []
                data_dict[key].append(value)

    return data_dict


async def rename_dataset(dataset):
    dataset = dataset.rename_column('ClientId', 'speaker_id')
    dataset = dataset.rename_column('Client_name', 'speaker_name')
    dataset = dataset.rename_column('Age', 'age')
    dataset = dataset.rename_column('Accent', 'accent')
    dataset = dataset.rename_column('Language', 'language')
    dataset = dataset.rename_column('Text', 'text')
    dataset = dataset.rename_column('Audiopath', 'audiopath')
    dataset = dataset.rename_column('Gender', 'gender')
    return dataset

async def remove_dataset(dataset):
    dataset = dataset.remove_columns(['language', 'age'])
    return dataset


# Function to yield batches of a specified size from a dataset
def batched_data(dataset, batch_size):
    dataset = dataset.to_dict()  # Convert to dictionary for easier batch slicing
    keys = list(dataset.keys())
    total_length = len(dataset[keys[0]])
    for i in range(0, total_length, batch_size):
        yield {key: dataset[key][i:i + batch_size] for key in keys}



# Function to download audio file from URL
def download_audio(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:

            f.write(response.content)
        return True
    else:
        print(f"Failed to download {url}")
        return False

# Function to load audio data and extract information
# def load_audio_info(audiopath):
#     print("ERROR Started")
#     audio_dict = {}
#     # Download audio file
#     local_filename = os.path.basename(audiopath)
#     if download_audio(audiopath, local_filename):
#         time.sleep(10)
#         # Load audio using librosa
#         audio_data, sampling_rate = librosa.load(local_filename, sr=None)
#         # Add audio information to the dictionary
#         audio_dict["path"] = local_filename
#         audio_dict["array"] = audio_data
#         audio_dict["sampling_rate"] = sampling_rate
#         # Delete the downloaded file
#         os.remove(local_filename)
#     else:
#         print(f"Failed to process {audiopath}")
#     print("ERROR End")
    
#     return audio_dict

from pydub import AudioSegment

def load_audio_info(audiopath):
    audio_dict = {}
    local_filename = os.path.basename(audiopath)
    if download_audio(audiopath, local_filename):
        time.sleep(1)
        try:
            audio = AudioSegment.from_file(local_filename)
            audio_data = np.array(audio.get_array_of_samples())
            sampling_rate = audio.frame_rate
        except Exception as e:
            print(f"Error reading audio file with pydub: {e}")
            return None

        audio_dict["path"] = local_filename
        audio_dict["array"] = audio_data
        audio_dict["sampling_rate"] = sampling_rate
        os.remove(local_filename)
    else:
        print(f"Failed to process {audiopath}")
    return audio_dict



import re
import unicodedata
from datetime import datetime
# Mapping dictionary for converting numbers to Dutch text
english_numbers = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine"
}
dutch_numbers = {
     "0": "nul",
  "1": "één",
  "2": "twee",
  "3": "drie",
  "4": "vier",
  "5": "vijf",
  "6": "zes",
  "7": "zeven",
  "8": "acht",
  "9": "negen",
  "10": "tien",
  "11": "elf",
  "12": "twaalf",
  "13": "dertien",
  "14": "veertien",
  "15": "vijftien",
  "16": "zestien",
  "17": "zeventien",
  "18": "achttien",
  "19": "negentien",
  "20": "twintig",
  "21": "éénentwintig",
  "22": "tweeëntwintig",
  "23": "drieëntwintig",
  "24": "vierentwintig",
  "25": "vijfentwintig",
  "26": "zesentwintig",
  "27": "zevenentwintig",
  "28": "achtentwintig",
  "29": "negenentwintig",
  "30": "dertig",
  "31": "éénendertig",
  "32": "tweeëndertig",
  "33": "drieëndertig",
  "34": "vierendertig",
  "35": "vijfendertig",
  "36": "zesendertig",
  "37": "zevenendertig",
  "38": "achtendertig",
  "39": "negenendertig",
  "40": "veertig",
  "41": "éénenveertig",
  "42": "tweeënveertig",
  "43": "drieënveertig",
  "44": "vierenveertig",
  "45": "vijfenveertig",
  "46": "zesenveertig",
  "47": "zevenenveertig",
  "48": "achtenveertig",
  "49": "negenenveertig",
  "50": "vijftig",
  "51": "éénenvijftig",
  "52": "tweeënvijftig",
  "53": "drieënvijftig",
  "54": "vierenvijftig",
  "55": "vijfenvijftig",
  "56": "zesenvijftig",
  "57": "zevenenvijftig",
  "58": "achtenvijftig",
  "59": "negenenvijftig",
  "60": "zestig",
  "61": "éénenzestig",
  "62": "tweeënzestig",
  "63": "drieënzestig",
  "64": "vierenzestig",
  "65": "vijfenzestig",
  "66": "zesenzestig",
  "67": "zevenenzestig",
  "68": "achtenzestig",
  "69": "negenenzestig",
  "70": "zeventig",
  "71": "éénenzeventig",
  "72": "tweeënzeventig",
  "73": "drieënzeventig",
  "74": "vierenzeventig",
  "75": "vijfenzeventig",
  "76": "zesenzeventig",
  "77": "zevenenzeventig",
  "78": "achtenzeventig",
  "79": "negenenzeventig",
  "80": "tachtig",
  "81": "éénentachtig",
  "82": "tweeëntachtig",
  "83": "drieëntachtig",
  "84": "vierentachtig",
  "85": "vijfentachtig",
  "86": "zesentachtig",
  "87": "zevenentachtig",
  "88": "achtentachtig",
  "89": "negenentachtig",
  "90": "negentig",
  "91": "éénennegentig",
  "92": "tweeënnegentig",
  "93": "drieënnegentig",
  "94": "vierennegentig",
  "95": "vijfennegentig",
  "96": "zesennegentig",
  "97": "zevenennegentig",
  "98": "achtennegentig",
  "99": "negenennegentig",
  "100": "honderd"
}

def convert_numbers_to_dutch(text):
    converted_text = ""
    for char in text:
        if char.isdigit():
            converted_text += " ".join(dutch_numbers.get(digit, digit) for digit in char) + " "
        else:
            converted_text += char
    return converted_text.strip()

def add_normalize_text(example):
    text = example['text']
    # Remove non-ASCII characters
    # text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Convert numbers to Dutch text with digits separated
    text = convert_numbers_to_dutch(text)
    # Capitalize the character following a comma (",") or period (".")
    text = re.sub(r'(?<=[,.])\s*([a-zA-Z])', lambda match: match.group(0).upper(), text)
    # Remove special characters except for '@'
    text = re.sub(r'[^@\w\s]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Capitalize the first word
    text = text.capitalize()
    # Check if text ends with ".", "?", or ","
    if not text.endswith((".", "?", ",")):
        # Add "." at the end
        text += "."
    example['normalized_text'] = text
    print(text)
    return example



def update_speaker_ids(dataset, column_name,value_for_column,make_speaker_id_same):
    unique_speaker_ids = set(dataset['train'][column_name])
    if make_speaker_id_same:
        id_mapping = {speaker_id:value_for_column for i, speaker_id in enumerate(unique_speaker_ids)}
    else:
        id_mapping = {speaker_id:str(i+1) for i, speaker_id in enumerate(unique_speaker_ids)}


    # Define a function to update the speaker_id values
    def update_example(example):
        example[column_name] = id_mapping[example[column_name]]
        return example

    concatenated_dataset = dataset.map(update_example)
    return concatenated_dataset



def filter_dataset(dataset,config):
    if config["should_filter"]:

            #filter according to Gender type list passed in list
            if config["should_filter_gender"]:
                try:
                    if config["should_filter_gender_param"]:
                        dataset = dataset.filter(lambda x: x['gender'] in config["should_filter_gender_param"])
                    else:
                        raise ValueError("If you set 'should_filter_gender' TRUE, then you must specify the 'should_filter_gender_param' in config.json as an string array.")
                except (KeyError, ValueError) as e:
                    print("Error:", e)
            #filter according to Accent type list passed in list
            if config["should_filter_accent"]:
                try:
                    if config["should_filter_accent_param"]:
                        dataset = dataset.filter(lambda x: x['accent'] in config["should_filter_accent_param"])
                    else:
                        raise ValueError("If you set 'should_filter_accent' TRUE, then you must specify the should_filter_accent_param in config.json as an string array.")
                except (KeyError, ValueError) as e:
                    print("Error:", e)
            
            
            #filter according to Language type list passed in list
            if config["should_filter_language"]:
                try:
                    if config["should_filter_language_param"]:
                        dataset = dataset.filter(lambda x: x['language'] in config["should_filter_language_param"])
                    else:
                        raise ValueError("If you set 'should_filter_language' TRUE, then you must specify the 'should_filter_language_param' in config.json as an string array.")
                except (KeyError, ValueError) as e:
                    print("Error:", e)
                
            #filter according to Client name list passed in list
            if config["should_filter_speaker_name"]:
                try:
                    if config["should_filter_speaker_name_param"]:
                        dataset = dataset.filter(lambda x: x['speaker_name'] in config["should_filter_speaker_name_param"])
                    else:
                        raise ValueError("If you set 'should_filter_speaker_name' TRUE, then you must specify the 'should_filter_speaker_name_param' in config.json as an string array.")
                except (KeyError, ValueError) as e:
                    print("Error:", e)
                

            #filter text containing specific word  
            if config["should_filter_sentence_containing_word"]:
                try:
                    if config["should_filter_sentence_containing_word_param"]:
                        dataset = dataset.filter(lambda x: any(word in x['text'] for word in config["should_filter_sentence_containing_word_param"]))
                    else:
                        raise ValueError("If you set 'should_filter_sentence_containing_word' TRUE, then you must specify the 'should_filter_sentence_containing_word_param' in config.json as an string array.")
                except (KeyError, ValueError) as e:
                    print("Error:", e)


    return dataset

async def main():
    args = parse_args()
    config = load_config(args.config)

    try:
        login(token=config["huggingface_hub_token"])
    except Exception as e:
        print("Error logging into hub:", str(e))
    
    if not config["concatenate_only_existings_dataset"]:
        data_dict = fetch_data_from_datacreation_tool(config["generate_specific_new_datas"],config["start_end_value"])
        if not data_dict:
            return
    
        dataset = Dataset.from_dict(data_dict)
        # dataset = dataset.select(range(4))
        dataset = await rename_dataset(dataset) #rename and remove non-required columns
        dataset = filter_dataset(dataset,config)
        dataset = await remove_dataset(dataset) #remove non-required columns
        
        time.sleep(10)
        list_of_batches = []
        batch_size = int(config["batch_size"])
        for i, batch_dict in enumerate(batched_data(dataset, batch_size)):
            batch = Dataset.from_dict(batch_dict)
            dataset_dict = DatasetDict({
                'train': batch
            })
            list_of_batches.append(dataset_dict)
            time.sleep(2)
        
        datasets_list = []
        for batch in list_of_batches:
            audio_path_list = batch['train']['audiopath']
            audio_dicts = []
            for audiopath in audio_path_list:
                audio_dict = load_audio_info(audiopath)
                if audio_dict:
                    audio_dicts.append(audio_dict)

            #add audio column to dataset
            assert len(audio_dicts) == len(batch['train']['audiopath']) #"The length of the audio data list must match the dataset length."

            # Specify the feature type for the audio column
            batch['train'] = batch['train'].add_column(name="audio",column=audio_dicts)
            features = batch['train'].features.copy()  # Copy existing features
            features['audio'] = Audio()  # Add an Audio feature for the new audio column

            # Create a new dictionary for the updated data
            updated_data = batch['train'].to_dict()
            updated_data['audio'] = audio_dicts  # Add the audio column


            # Create a new dataset with the updated data and defined features
            updated_dataset = batch['train'].from_dict(updated_data, features=features)
            updated_dataset = updated_dataset.remove_columns('audiopath')
            
            updated_dataset = updated_dataset.map(add_normalize_text)
            updated_dataset_train = DatasetDict({
                'train': updated_dataset
            })
            datasets_list.append(updated_dataset_train)
        
        #concatenaed all batched datset
        updated_datasets_to_concatenate = []
        for i in range(len(datasets_list)):
            updated_datasets_to_concatenate.append(datasets_list[i]['train'])

        # Concatenate the datasets
        updated_concatenated_dataset = concatenate_datasets(updated_datasets_to_concatenate)
        updated_generated_dataset_train = DatasetDict({
                'train': updated_concatenated_dataset
            })

        loaded_datasets = []
        total_existing_dataset = config["existing_dataset_names"]

        if config["concatenate_existing_dataset_with_above_generate_dataset"]:
            if config["existing_dataset_names"]:
                
                for repo_id in total_existing_dataset:
                    dataset = load_dataset(repo_id)
                    loaded_datasets.append(dataset)
                    print("Loaded  Dataset from existing only  is:",loaded_datasets)
                loaded_datasets.append(updated_generated_dataset_train)
                print("Loaded  Dataset is:",loaded_datasets)
                concatenated_dataset = concatenate_datasets([d['train'] for d in loaded_datasets])
                print("Cconcatenated Dataset is:",concatenated_dataset)
                updated_generated_dataset_train = DatasetDict({
                    'train': concatenated_dataset
                })
                updated_generated_dataset_train = filter_dataset(updated_generated_dataset_train,config)

            else:
                updated_generated_dataset_train = filter_dataset(updated_generated_dataset_train,config)
        else:
            updated_generated_dataset_train = filter_dataset(updated_generated_dataset_train,config)
            
    else:
        try:
            loaded_datasets = []
            total_existing_dataset = config["existing_dataset_names"]
            for repo_id in total_existing_dataset:
                    dataset = load_dataset(repo_id)
                    loaded_datasets.append(dataset)
            concatenated_dataset = concatenate_datasets([d['train'] for d in loaded_datasets])
            updated_generated_dataset_train = DatasetDict({
                    'train': concatenated_dataset
                })
            updated_generated_dataset_train = filter_dataset(updated_generated_dataset_train,config)
            
        except KeyError:
            print("Key 'existing_dataset_names' in the config.json may not available in huggingface")


    if config["make_change_on_columns"]:
        
        if config["make_speaker_id_same"]:
            concatenated_dataset = update_speaker_ids(updated_generated_dataset_train, 'speaker_id',str(config["make_speaker_id_same_value"]),config["make_speaker_id_same"])
            concatenated_dataset.push_to_hub(repo_id=config["dataset_name"])
        else:
            distinct_speaker_names = set(updated_generated_dataset_train['train']['speaker_name'])
            for index, speaker in enumerate(distinct_speaker_names):
                concatenated_dataset = update_speaker_ids(updated_generated_dataset_train, 'speaker_id', index,config["make_speaker_id_same"])
                concatenated_dataset.push_to_hub(repo_id=config["dataset_name"])


    else:
        updated_generated_dataset_train.push_to_hub(repo_id=config["dataset_name"])


    

if __name__ == "__main__":
    asyncio.run(main())




