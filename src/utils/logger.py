import logging
import sys
from logging.handlers import RotatingFileHandler

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    
    # Only add handlers if the logger doesn't have them (prevents duplicate logs)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # 1. Create formatters
        common_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 2. Console Handler (Shows INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(common_format)

        # 3. File Handler (Stores everything, rotates at 5MB)
        file_handler = RotatingFileHandler(
            'app.log', maxBytes=5*1024*1024, backupCount=2
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(common_format)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger