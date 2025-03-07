import logging

# 로그 설정
logging.basicConfig(
    level=logging.DEBUG,  # 출력할 로그의 레벨 설정
    format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 출력 형식
    datefmt="%Y-%m-%d %H:%M:%S",  # 날짜 형식
)

# 로그 출력
logging.debug("이것은 디버그 메시지입니다.")
logging.info("이것은 정보 메시지입니다.")
logging.warning("이것은 경고 메시지입니다.")
logging.error("이것은 에러 메시지입니다.")
logging.critical("이것은 치명적인 에러 메시지입니다.")
