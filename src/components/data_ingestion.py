import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.model_trainer import ModelTrainer


## intialize the data ingestion configuration

@dataclass
class DataIngestionconfig:
    train_set_path=os.path.join('artifacts','train_set.csv')
    test_set_path=os.path.join('artifacts','test_set.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')


## create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            # Importing the dataset 
            data = pd.read_csv(r'A:\Analytix\Machine learning\ML projects\Demand Estimation\Data\Electricity Consumption.csv')

            ## Converting Date column to datetime & setting it as index
            data['DATE'] = pd.to_datetime(data.DATE,format='%m/%d/%Y')
            data = data.set_index(data.DATE)
            data_f = data['Electricty_Consumption_in_TW']


            if data_f.isna().any().any():
                data = data_f.apply(lambda x: x.fillna(x.rolling(window=3, min_periods=1).mean()))

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Train test split")

            size = int(len(data)*0.95)

            ## Train Test split
            train = data[:size]
            test = data[size:]

            train.to_csv(self.ingestion_config.train_set_path,index=False,header=True)
            test.to_csv(self.ingestion_config.test_set_path,index=False,header=True)

            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_set_path,
                self.ingestion_config.test_set_path
            )
        except Exception as e:
            raise CustomException(e,sys)