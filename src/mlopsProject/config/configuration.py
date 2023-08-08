from pathlib import Path
import os
import sys
import pandas as pd
from mlopsProject.utils.common import read_yaml, create_directories, save_object
from mlopsProject.entity.config_entity import DataIngestionConfig, DataTransformationConfig
from mlopsProject import logger as Logger
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline




class DataIngestion:
    def __init__(
        self):
        self.ingestion_config = DataIngestionConfig()
        self.transformation_config = DataTransformationConfig()

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
            num_col = ['writing_score', 'reading score']
            cat_col = ['gender',
             'race/ethnicity', 
             'parental level of education', 
             'lunch','test preparation course']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler())
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
                test_arr,
                self.transformation_config.preprocessor,


            )

        except Exception as e:
            Logger.error(f">>>>>> stage failed <<<<<<<")
            Logger.error(f">>>>>>> {e} <<<<<<<")
        



            

        