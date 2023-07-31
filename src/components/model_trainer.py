# Basic Import
import numpy as np
import pickle
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error


from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainer:
    def initate_model_training(self,train_set,test_set):
        try:
            model = SARIMAX(train_set, order = (2,0,2), seasonal_order= (2,2,2,12),
                                             enforce_stationarity=False,
                                             enforce_invertibility=False).fit()
            
            fcst = model.forecast(steps=int(len(test_set)))

            # Mean Absolute Error (MAE)
            mae = mean_absolute_error(test_set,fcst)

            # Root Mean Squared Error (RMSE)
            rmse = mean_squared_error(test_set, fcst, squared=False)

            # Coefficient of Determination (R-squared)
            r2 = r2_score(test_set,fcst)

            #Calculating MAPE
            MAPE = mean_absolute_percentage_error(test_set,fcst)

            # Print the metrics
            print("Mean Absolute Error (MAE):", mae)
            print("Root Mean Squared Error (RMSE):", rmse)
            #print("Coefficient of Determination (R-squared):", r2)
            print("Mean Absolute Percentage Error:", MAPE)

            logging.info("Model report printed")

            path = r'A:\Analytix\Machine learning\ML projects\Demand Estimation\artifacts\model.pkl'


            # saving best model into a picke file
            with open(path, 'wb') as f:
                pickle.dump(model, f)

            

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)