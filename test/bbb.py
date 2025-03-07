import logging

# Logger 생성
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)  # 로그 레벨 설정

# 콘솔 핸들러 추가
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 포맷 지정
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Logger에 핸들러 추가
logger.addHandler(console_handler)

# 로그 출력
logger.debug("디버그 메시지")
logger.info("정보 메시지")
logger.warning("경고 메시지")
logger.error("에러 메시지")
logger.critical("치명적인 에러 메시지")