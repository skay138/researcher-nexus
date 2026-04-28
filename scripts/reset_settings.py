"""
MariaDB system_config 테이블의 값을 기본값(CONFIG_DEFAULTS)으로 초기화하는 스크립트.
기존에 변경된 설정이 있더라도 모두 무시하고 소스코드의 기본값으로 덮어씁니다.
"""

import json
import logging
import sys
import os

# 프로젝트 루트 내 src 디렉토리를 PYTHONPATH에 추가
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_dir, "src"))

from common.config.settings import get_settings
from common.config.query_config import CONFIG_DEFAULTS
from infrastructure.mariadb import _parse_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def reset_settings():
    import pymysql
    
    settings = get_settings()
    mariadb_url = settings.mariadb_url
    params = _parse_url(mariadb_url)
    
    logger.info("MariaDB 설정(system_config) 초기화 시작...")
    
    try:
        conn = pymysql.connect(**params)
        with conn.cursor() as cur:
            # 1. 기존 설정 삭제 (선택 사항: 전체 삭제 후 재삽입 또는 REPLACE INTO)
            # 여기서는 안전하게 전체 삭제 후 다시 넣는 방식을 취합니다.
            cur.execute("DELETE FROM system_config")
            logger.info("기존 설정 삭제 완료")
            
            # 2. 기본값 삽입
            for key, value in CONFIG_DEFAULTS.items():
                cur.execute(
                    "INSERT INTO system_config (`key`, `value`) VALUES (%s, %s)",
                    (key, json.dumps(value))
                )
                logger.info(f"설정 적용: {key} = {value}")
                
        conn.commit()
        conn.close()
        logger.info("MariaDB 설정 초기화 완료.")
        
    except Exception as e:
        logger.error(f"설정 초기화 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    reset_settings()
