import mlops-project
from pathlib import Path
import os
from mlops_project.entity import DataIngestionConfig
from src.logger import Logger


class ConfigurationManager:
    def __init__(
        self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> DataIngestionConfig:
        Logger.info(f"Initiating data ingestion method or component")
        try:
            self.ingestion_config = DataIngestionConfig()
        except Exception as e:
            Logger.error(f"Error while initiating data ingestion method or component: {e}")