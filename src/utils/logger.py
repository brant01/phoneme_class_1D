import logging
from pathlib import Path
from datetime import datetime

def create_logger(log_dir: Path) -> logging.Logger:
    """
    Create a logger that writes to both console and a timestamped log file.

    Args:
        log_dir (Path): Directory where log file will be stored.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    logger = logging.getLogger("phoneme_project")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Ensure clean handler state

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
