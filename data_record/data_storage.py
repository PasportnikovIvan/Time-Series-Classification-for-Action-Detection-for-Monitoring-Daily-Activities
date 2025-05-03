#data_record/data_storage.py
# # Module for data storage and management

import json

def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)