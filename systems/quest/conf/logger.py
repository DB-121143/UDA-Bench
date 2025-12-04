
import os

class QUESTLogger:
    def __init__(self, path, log_file_name, mode="a"):
        self.path = path
        self.log_file_name = log_file_name
        self.mode = mode
        os.makedirs(path, exist_ok=True)
        self.file_pointer = open(path + log_file_name, mode)

    def write(self, message):
        self.file_pointer.write(message)
        self.file_pointer.flush()

    def _close(self):
        self.file_pointer.close()

    def __del__(self):
        self._close()
