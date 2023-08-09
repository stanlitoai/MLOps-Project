from dataclasses import dataclass, field
import os

from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


@dataclass(frozen=True)
class DataTransformationConfig:
    preprocessor: str = os.path.join('artifacts', 'preprocessor.pkl')
    # test_data_path: str = os.path.join('artifacts', 'test.csv')
    # raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
    
@dataclass(frozen=True)
class ModelConfig:
    model_path: str = os.path.join('artifacts','model.pkl')
    
