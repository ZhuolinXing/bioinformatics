import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger('my_logger')
# 创建文件处理器
logger.setLevel(level=logging.INFO)
file_handler = RotatingFileHandler('log/log_file.log', maxBytes=10*1024*1024, backupCount=3)

# 配置文件处理器的格式
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)d  :  %(message)s'))

# 将文件处理器添加到日志记录器
logger.addHandler(file_handler)
logger.debug("log init success")
