from ..models.model_interface import LightGBMModel
from ..models.model_interface import RandomForestModel
from ..models.model_interface import XGBoostModel
from ..models.model_interface import ModelInterface
from typing import Optional, Dict, Any


class ModelFactory:
    """Factory class for creating model instances."""
    
    MODEL_TYPES = {
        'lightgbm': LightGBMModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, params: Optional[Dict[str, Any]] = None) -> ModelInterface:
        """Create a model instance of the specified type."""
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(cls.MODEL_TYPES.keys())}")
        
        return cls.MODEL_TYPES[model_type](params=params)
    
    @classmethod
    def deserialize_model(cls, model_type: str, data: bytes) -> ModelInterface:
        """Deserialize bytes into a model instance."""
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(cls.MODEL_TYPES.keys())}")
        
        return cls.MODEL_TYPES[model_type].deserialize(data)
    

