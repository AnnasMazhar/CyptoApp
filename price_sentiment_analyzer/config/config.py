# config.py
import yaml
import os
from typing import Any, Dict, Optional

class Config:
    """Configuration management class that loads from YAML files with environment support."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config_data = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load the configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        
        with open(self.config_path, 'r') as file:
            self.config_data = yaml.safe_load(file)
        
        # Apply environment-specific overrides if available
        env = os.environ.get("APP_ENV", "development")
        if env in self.config_data.get("environments", {}):
            env_config = self.config_data["environments"][env]
            self._merge_configs(self.config_data, env_config)
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge override config into base config."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key with optional default."""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def validate(self, schema: Dict[str, Any]) -> bool:
        """Validate configuration against a schema."""
        # Basic validation implementation - could be enhanced with a schema validation library
        for key, required in schema.items():
            if required and self.get(key) is None:
                return False
        return True
    
