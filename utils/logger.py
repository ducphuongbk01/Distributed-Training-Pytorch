import os
import logging


class Logger(object):
    def __init__(self, log_name, file):
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)

        # handler
        if os.path.exists(file):
            os.remove(file)

        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(file)

        # add formater
        form = logging.Formatter(fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                 datefmt = '%Y-%m-%d   %H:%M:%S')
        stream_handler.setFormatter(form)
        file_handler.setFormatter(form)

        # add handler
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    def log(self, message, log_type):
        if log_type=="warning":
            self.logger.warning(message)
        elif log_type=="error":
            self.logger.error(message)
        else:
            self.logger.info(message)
