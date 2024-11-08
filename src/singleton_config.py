
from src.config import Configuration
from src.data_classes import dataclass
from typing import Dict, Any  


# @dataclass
# class RerankingConfig:
#     enabled: bool
#     provider: str
#     models: Dict[str, Any]
#     thresholds: Dict[str, float]
#     active_model: str

    
class ConfigSingleton:
    _instance = None

    def __new__(cls, config_file='config.yaml'):
        if cls._instance is None:
            cls._instance = Configuration(config_file)
        return cls._instance



    def get_path_config(self, key, default=None):
        return self.config.get('paths', {}).get(key, default)