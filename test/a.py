import logging

logger = logging.getLogger(__name__)  # Logger instance 생성

logging.basicConfig(
    level=logging.DEBUG,  # DEBUG부터 모든 로그를 출력하도록 설정
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%m-%d-%Y %H:%M:%S',  # 날짜 형식 수정
    # filename='example.log',  # 파일 로그가 필요하면 주석 해제
    # encoding='utf-8',
)

logger.debug('This message should go to the log file')  # 출력됨
logger.info('So should this')  # 출력됨
logger.warning('And this, too')  # 출력됨
logger.error('And non-ASCII stuff, too, like Øresund and Malmö')  # 출력됨
logger.critical("Critical")  # 출력됨