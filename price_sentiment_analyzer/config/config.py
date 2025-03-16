import yaml
from pydantic import BaseModel, ValidationError
from pathlib import Path
from typing import Dict, Any

class Settings(BaseModel):
    database: Dict[str, Any]
    features: Dict[str, Any]
    models: Dict[str, Any]
    logging: Dict[str, Any]

class Config:
    def __init__(self, env: str = 'dev'):
        self.env = env
        self.settings = self._load_config()
        
    def _load_config(self) -> Settings:
        config_path = Path(__file__).parent / f"config_{self.env}.yaml"
        try:
            with open(config_path) as f:
                raw_config = yaml.safe_load(f)
                return Settings(**raw_config)
        except (FileNotFoundError, ValidationError) as e:
            raise RuntimeError(f"Config error: {str(e)}")

# Example config_dev.yaml
"""
database:
  filename: crypto_data.db
  pool_size: 5
features:
  lag_periods: [1, 3, 5]
  ta_indicators:
    rsi_period: 14
    macd_fast: 12
models:
  default: lightgbm
  params:
    lightgbm:
      num_leaves: 31
      learning_rate: 0.05
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""