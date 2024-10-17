
from src.config import Configuration

class ConfigSingleton:
    _instance = None

    def __new__(cls, config_file='config.yaml'):
        if cls._instance is None:
            cls._instance = Configuration(config_file)
        return cls._instance



    def get_path_config(self, key, default=None):
        return self.config.get('paths', {}).get(key, default)