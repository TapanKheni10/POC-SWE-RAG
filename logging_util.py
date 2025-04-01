import os
import json
import logging
from datetime import datetime

class JSONFormatter(logging.Formatter):
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "levelname": record.levelname,
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno
        }

        message = record.getMessage()
        try:
            parsed_message = json.loads(message)
            log_entry["message"] = json.dumps(parsed_message, indent=4, ensure_ascii=False)
        except json.JSONDecodeError:
            log_entry["message"] = message
        
        if record.args:
            log_entry["extra"] = record.args
            
        return json.dumps(log_entry, ensure_ascii = False, indent=4)

def setup_logger(name: str, log_file: str, log_dir: str = "struct_logs", level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger with a specified name and log file.
    
    Args:
        name (str): The name of the logger.
        log_file (str): The name of the log file.
        log_dir (str): Directory where logs will be stored.
        level (int): Logging level (default: logging.INFO).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.FileHandler(log_path)
    handler.setFormatter(JSONFormatter())

    logger.addHandler(handler)
    return logger

loggers = {
    "ChunkLogger" : setup_logger("chunker", "chunking.log"),
    "GeminiLogger" : setup_logger("gemini", "gemini_usage.log"),
    "VoyageLogger" : setup_logger("voyageai", "embedding_generation.log"),
    "PineconeLogger" : setup_logger("pinecone", "pinecone_usage.log")
}