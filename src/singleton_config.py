
from src.config import Configuration

class ConfigSingleton:
    _instance = None

    def __new__(cls, config_file='config.yaml'):
        if cls._instance is None:
            cls._instance = Configuration(config_file)
        return cls._instance
