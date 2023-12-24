import logging
import os.path
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('my_logger')

def initlog(log_path :str):
    # 创建文件处理器
    logger.setLevel(level=logging.INFO)
    os.makedirs(log_path, mode=0o644, exist_ok=True)
    file_handler = RotatingFileHandler(os.path.join(log_path,'log.log'), maxBytes=10*1024*1024, backupCount=3)

    # 配置文件处理器的格式
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)d  :  %(message)s'))

    # 将文件处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.debug("log init success")
