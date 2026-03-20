# import logging
#
# # 配置日志器
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fhandler = logging.FileHandler('livetalking.log')  # 可以改为StreamHandler输出到控制台或多个Handler组合使用等。
# fhandler.setFormatter(formatter)
# fhandler.setLevel(logging.INFO)
# logger.addHandler(fhandler)
#
# # handler = logging.StreamHandler()
# # handler.setLevel(logging.DEBUG)
# # sformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# # handler.setFormatter(sformatter)
# # logger.addHandler(handler)







import logging
import sys

# 创建日志器
logger = logging.getLogger("livetalking")
logger.setLevel(logging.DEBUG)

# 日志格式
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ===== 文件日志（UTF-8，永久保存）=====
file_handler = logging.FileHandler(
    'livetalking.log',
    encoding='utf-8'   # 🔴 关键：防止中文和 emoji 崩溃
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# ===== 控制台日志（UTF-8，实时调试）=====
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# 🔴 强制控制台使用 UTF-8
console_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)

# 防止重复添加 handler
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
