import logging
import os
import quest.conf.settings as settings

# 创建 logger
logger = logging.getLogger("quest")
logger.setLevel(logging.DEBUG)

# 创建 formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 添加 handlers（避免重复添加）
if not logger.hasHandlers():
    # 控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    file_handler = logging.FileHandler(settings.LOG_DIR_NAME, mode='a', encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# ✅ 包含功能函数
def print_log(*args):
    msg = ' '.join(str(arg) for arg in args)
    msg = '\n' + msg
    logger.info(msg)

def log_custom_message(level, message):
    if level == "debug":
        logger.debug(message)
    elif level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)
    else:
        logger.info(f"[UNKNOWN LEVEL] {message}")