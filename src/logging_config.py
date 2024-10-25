import logging
import logging.config
import os
import psutil
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

class RAGDebugFilter(logging.Filter):
    """Filter for adding system info to debug messages"""
    def filter(self, record):
        process = psutil.Process(os.getpid())
        # Set these fields for all debug records
        record.memory_usage = f"{process.memory_info().rss / 1024 / 1024:.2f} MB"
        record.cpu_percent = f"{process.cpu_percent()}%"
        record.module_line = f"{record.module}:{record.lineno}"
        record.function_name = record.funcName
        record.component = getattr(record, 'component', 'general')
        record.operation = getattr(record, 'operation', 'general')
        return True

class RAGInfoFilter(logging.Filter):
    """Filter for standard pipeline operation messages"""
    def filter(self, record):
        # Ensure required fields are present
        record.component = getattr(record, 'component', 'pipeline')
        record.operation = getattr(record, 'operation', 'general')
        record.module_line = f"{record.module}:{record.lineno}"
        record.function_name = record.funcName
        return True

class RAGErrorFilter(logging.Filter):
    """Filter for error tracking and identification"""
    def filter(self, record):
        # Set error-specific fields and ensure required fields are present
        record.error_id = str(uuid.uuid4())[:8]
        record.component = getattr(record, 'component', 'unknown')
        record.operation = getattr(record, 'operation', 'error')
        record.module_line = f"{record.module}:{record.lineno}"
        record.function_name = record.funcName
        return True

def setup_rag_logging(log_dir: str = 'logs', unified_log: bool = False):
    """
    Setup logging configuration for the RAG pipeline
    Args:
        log_dir: Directory for log files
        unified_log: If True, also output all logs to a single combined file
    """
    
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create unified log filename with timestamp if enabled
    unified_log_file = None
    if unified_log:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unified_log_file = str(log_path / f'rag_unified_{timestamp}.log')

    handlers = {
        # Debug handlers
        'debug_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(log_path / 'rag_debug.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'debug',
            'filters': ['debug_filter'],
            'level': 'DEBUG'
        },
        'debug_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'debug',
            'filters': ['debug_filter'],
            'level': 'DEBUG'
        },
        # Info handlers
        'info_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(log_path / 'rag_info.log'),
            'maxBytes': 10485760,
            'backupCount': 5,
            'formatter': 'info',
            'filters': ['info_filter'],
            'level': 'INFO'
        },
        'info_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'info',
            'filters': ['info_filter'],
            'level': 'INFO'
        },
        # Error handlers
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(log_path / 'rag_error.log'),
            'maxBytes': 10485760,
            'backupCount': 5,
            'formatter': 'error',
            'filters': ['error_filter'],
            'level': 'ERROR'
        },
        'error_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'error',
            'filters': ['error_filter'],
            'level': 'ERROR'
        }
    }

    # Add unified log handler if enabled
    if unified_log:
        handlers['unified_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': unified_log_file,
            'maxBytes': 20971520,  # 20MB
            'backupCount': 5,
            'formatter': 'unified',
            'level': 'DEBUG'
        }

    # Base handler list for all loggers
    base_handlers = ['debug_file', 'debug_console',
                    'info_file', 'info_console',
                    'error_file', 'error_console']
    
    if unified_log:
        base_handlers.append('unified_file')

    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'debug': {
                'format': '%(asctime)s - DEBUG - Module:%(module_line)s - [%(component)s] - '
                         'Memory: %(memory_usage)s - CPU: %(cpu_percent)s - %(message)s'
            },
            'info': {
                'format': '%(asctime)s - INFO - Module:%(module_line)s - [%(component)s] - '
                         '%(operation)s - %(message)s'
            },
            'error': {
                'format': '%(asctime)s - ERROR ID: %(error_id)s - Module:%(module_line)s - [%(component)s] - '
                         '%(message)s\n'
                         'Function: %(function_name)s\n'
                         'Full Path: %(pathname)s'
            },
            'unified': {
                'format': '%(asctime)s - %(levelname)s - Module:%(module_line)s - [%(component)s] - '
                         '%(operation)s - %(message)s\n'
                         'Function: %(function_name)s\n'
                         '---'
            }
        },
        'filters': {
            'debug_filter': {
                '()': RAGDebugFilter
            },
            'info_filter': {
                '()': RAGInfoFilter
            },
            'error_filter': {
                '()': RAGErrorFilter
            }
        },
        'handlers': handlers,
        'loggers': {
            'rag': {  # Parent logger
                'handlers': base_handlers,
                'level': 'DEBUG',
                'propagate': False
            },
            'rag.pipeline': {  # RAGPipeline class
                'handlers': base_handlers,
                'level': 'DEBUG',
                'propagate': False
            },
            'rag.system': {  # RAGSystem class
                'handlers': base_handlers,
                'level': 'DEBUG',
                'propagate': False
            },
            'rag.generator': {  # Generator class
                'handlers': base_handlers,
                'level': 'DEBUG',
                'propagate': False
            },
            'rag.embedding.azure': {  # AzureOpenAIEmbeddingGenerator
                'handlers': base_handlers,
                'level': 'DEBUG',
                'propagate': False
            },
            'rag.embedding.huggingface': {  # HuggingFaceEmbeddingGenerator
                'handlers': base_handlers,
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }

    # Apply configuration
    logging.config.dictConfig(config)
    
    logger = logging.getLogger('rag')
    if unified_log:
        logger.info(f"Unified logging enabled. Writing to: {unified_log_file}")
    
    return logger

def get_logger(component: str = None):
    """Get a logger for a specific component"""
    if component:
        return logging.getLogger(f'rag.{component}')
    return logging.getLogger('rag')

# Line number demonstration
if __name__ == "__main__":
    # Setup logging with unified log enabled
    logger = setup_rag_logging(unified_log=True)
    
    # Test 1: Basic line number demonstration
    variable = "some value"                     # Line X
    logger.debug("Debug after variable")        # Line X+1: This will show this line number
    
    # Test 2: Function context
    def test_function():
        inside_var = 42                         # Line Y
        logger.info("Info inside function")     # Line Y+1: Will show this line number
        logger.debug("Debug in function")       # Line Y+2: Will show this line number
        return inside_var
    
    # Test 3: Consecutive logs
    logger.debug("First debug")                 # Will show this line number
    logger.info("First info")                   # Will show this line number
    logger.error("First error")                 # Will show this line number
    
    # Test 4: Call function to see line numbers in function context
    test_function()
    
    # Test 5: Logs with spacing
    logger.debug("Debug message 1")             # Will show this line number
    
    logger.debug("Debug message 2")             # Will show this line number, different from above
    
    # Test 6: Component specific logging
    system_logger = get_logger('system')
    system_logger.debug("System debug")         # Will show this line number
    system_logger.info("System info")           # Will show this line number
    system_logger.error("System error")         # Will show this line number