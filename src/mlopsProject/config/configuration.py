from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
from mlopsProject.utils.common import read_yaml, create_directories, save_object, evaluate_models
from mlopsProject.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelConfig
from mlopsProject import logger as Logger
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Modelling GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
import warnings




class DataIngestion:
    def __init__(
        self):
        self.ingestion_config = DataIngestionConfig()
        self.transformation_config = DataTransformationConfig()
        self.model_config = ModelConfig()

    def initiate_data_ingestion(self):
        Logger.info(f"Initiating data ingestion method or component")
        try:
            df = pd.read_csv("research/data/student.csv")
            df.head()
            Logger.info(f"Successfully read data from csv file")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            Logger.info(f"Successfully created directories for train and test data")
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            Logger.info(f"Successfully wrote data to csv file")
            Logger.info(f"Train Test Split Initiated")

            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(self.ingestion_config.train_data_path ,index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path,index=False, header=True)
            Logger.info(f"Successfully wrote data to csv file for train test split")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
                    )
        

        except Exception as e:
            Logger.error(f">>>>>> stage failed <<<<<<<")
            Logger.error(f">>>>>> {e} <<<<<<<")

    def get_data_transformation(self):
        try:
            # num_col = X.select_dtypes(exclude="object").columns
            # cat_col = X.select_dtypes(include="object").columns
            num_col = ['writing score', 'reading score']
            cat_col = ['gender',
             'race/ethnicity', 
             'parental level of education', 
             'lunch','test preparation course']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            Logger.info(f"Category columns encoded successfully")

            Logger.info(f"Numerical columns encoded successfully")

            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, num_col),
                ('cat', cat_pipeline, cat_col)           ])


            return preprocessor

        except Exception as e:
            Logger.error(f">>>>>> stage failed <<<<<<<")
            Logger.error(f">>>>>> {e} <<<<<<<")


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)

            Logger.info(f"Successfully read data from csv file")
            Logger.info(f"Processing train data")

            preprocessing_obj = self.get_data_transformation()

            target_col = "math score"

            num_col = ['writing_score', 'reading score']

            input_features_train_df = train.drop(target_col, axis=1)
            target_features_train_df = train[target_col]

            input_features_test_df = test.drop(target_col, axis=1)
            target_features_test_df = test[target_col]
            Logger.info(f"Successfully processed {input_features_test_df}train and test data")

            Logger.info(f"Apply  processing  object  on  training data and test data")

            preprocessed_train_df = preprocessing_obj.fit_transform(input_features_train_df)
            preprocessed_test_df = preprocessing_obj.transform(input_features_test_df)

            Logger.info(f"Successfully processed train and test data")

            train_arr = np.c_[
                preprocessed_train_df,
                np.array(target_features_train_df)
            ]
            test_arr = np.c_[
                preprocessed_test_df,
                np.array(target_features_test_df)
            ]
            Logger.info(f"Successfully saved processing object")


            #os.makedirs(os.path.dirname(self.transformation_config.train_data_path), exist_ok=True)
            save_object(
                file_path = self.transformation_config.preprocessor,
                obj = preprocessing_obj
                )

            return (
                train_arr,
                test_arr


            )
            Logger.info(">>>>>>>>>Completed<<<<<<<<<<<<<<<<<")

        except Exception as e:
            Logger.error(f">>>>>> stage failed <<<<<<<")
            Logger.error(f">>>>>>> {e} <<<<<<<")
            
            
    def initiate_model_trainer(self,train_array,test_array):
        try:
            Logger.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            Logger.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_config.model_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise e
        



            

        