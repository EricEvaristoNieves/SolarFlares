# Import necessary libraries
import pandas as pd
from ucimlrepo import fetch_ucirepo 
import yaml
import argparse
from typing import Text
import os,sys

# Agregar la ruta del directorio padre al sys.path
# Agregar la ruta del directorio padre al sys.path
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_proyecto = os.path.dirname(os.path.dirname(ruta_actual))
sys.path.append(ruta_proyecto)

from src.utils.logs import get_logger

#preparacion de los datos
def data_load(config_path: Text):
   
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    #get info     
    dataset = config["data_load"]["dataset_id"]
    logger = get_logger('DATA_LOAD', log_level=config['base']['log_level'])
           
    solar_flare = fetch_ucirepo(id=dataset)     
    X = solar_flare.data.features 
    y = solar_flare.data.targets 
    data = pd.concat([X, y], axis=1, join="inner")

    data['moderate flares'] = data['moderate flares'].apply(lambda x: x * 10)
    data['severe flares'] = data['severe flares'].apply(lambda x: x * 20) 
    
    columns_to_keep = config["featurize"]["cols_to_keep"]
    cols_value = config["featurize"]["cols_value"]
    dataPrepared  = pd.melt(data, id_vars=columns_to_keep, value_vars=cols_value, var_name='flares', value_name='flares_value')

    dataPrepared.to_csv(config["data_load"]["dataset_procesed"] , index=False)
    logger.info('Save processed data')

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()    
    parametro = ruta_proyecto + args.config    
    data_load(config_path=parametro)
    #data_load()