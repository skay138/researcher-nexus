"""
Seed fixtures — SEED_NODES / SEED_RELATIONS
4개 도메인: 해양, AI/바이오 신약, 우주항공, 수소/재생에너지

사용처:
  - scripts/seed_data.py   (Neo4j / Milvus 초기 데이터)
  - data/in_memory.py      (USE_MOCK=true 인메모리 어댑터)
  - tests/test_architecture.py
"""

from __future__ import annotations
from typing import Dict, List, Any

# ─────────────────────────────────────────────────────────────────────────────
# NODES  {id → props}
# props 필수: id, type, name
# props 선택: year, text/abstract/summary, topic, expertise, keywords, ...
# ─────────────────────────────────────────────────────────────────────────────

SEED_NODES: Dict[str, Dict[str, Any]] = {

    # ══════════════════════════════════════════════════════════════════════════
    # 도메인 1: 해양 (Maritime)
    # ══════════════════════════════════════════════════════════════════════════

    # Organizations
    "org_kaist_ocean": {
        "id": "org_kaist_ocean", "type": "Organization",
        "name": "KAIST 해양시스템공학과",
        "text": "한국과학기술원 해양시스템 및 조선공학 전문 연구기관",
    },
    "org_kriso": {
        "id": "org_kriso", "type": "Organization",
        "name": "한국해양과학기술원 KRISO",
        "text": "선박 및 해양구조물 연구개발 전문 정부출연연구원",
    },
    "org_samsung_heavy": {
        "id": "org_samsung_heavy", "type": "Organization",
        "name": "삼성중공업",
        "text": "초대형 컨테이너선 LNG선 해양플랜트 건조 전문 조선사",
    },

    # Researchers
    "r_kim_ocean": {
        "id": "r_kim_ocean", "type": "Researcher",
        "name": "김해양",
        "expertise": "선박 유체역학 프로펠러 설계 해양 에너지 저감",
        "topic": "해양",
    },
    "r_lee_maritime": {
        "id": "r_lee_maritime", "type": "Researcher",
        "name": "이해운",
        "expertise": "자율운항선박 IoT 항법 시스템 선박 안전",
        "topic": "해양 자율운항",
    },
    "r_park_ocean": {
        "id": "r_park_ocean", "type": "Researcher",
        "name": "박조선",
        "expertise": "LNG 추진 시스템 그린십 온실가스 저감",
        "topic": "해양 친환경",
    },

    # Projects
    "proj_autonomous_ship": {
        "id": "proj_autonomous_ship", "type": "Project",
        "name": "자율운항선박 핵심기술 개발",
        "year": 2023,
        "text": "인공지능 기반 자율운항 및 원격 모니터링 기술을 적용한 스마트 선박 개발 프로젝트",
        "topic": "자율운항 AI 항법",
    },
    "proj_green_ship": {
        "id": "proj_green_ship", "type": "Project",
        "name": "그린십 친환경 추진 기술",
        "year": 2022,
        "text": "LNG·암모니아 이중연료 추진 시스템 및 에너지 효율 최적화 기술 개발",
        "topic": "그린십 LNG 친환경",
    },

    # Papers
    "paper_ship_ai": {
        "id": "paper_ship_ai", "type": "Paper",
        "name": "딥러닝 기반 선박 충돌 회피 알고리즘",
        "year": 2023,
        "abstract": "강화학습과 LSTM을 결합한 선박 자율운항 충돌 회피 시스템. COLREGs 준수 알고리즘 포함.",
        "keywords": "자율운항 충돌회피 강화학습 선박",
    },
    "paper_lng_propulsion": {
        "id": "paper_lng_propulsion", "type": "Paper",
        "name": "LNG 이중연료 엔진 성능 최적화 연구",
        "year": 2022,
        "abstract": "LNG와 HFO 이중연료 엔진 연소 특성 분석 및 NOx·SOx 배출 저감 최적화.",
        "keywords": "LNG 이중연료 엔진 배출가스",
    },

    # Patents
    "patent_ship_nav": {
        "id": "patent_ship_nav", "type": "Patent",
        "name": "AI 기반 선박 경로 최적화 장치 및 방법",
        "year": 2023,
        "summary": "딥러닝 모델을 활용한 실시간 항로 최적화 및 장애물 회피 특허",
        "patent_number": "KR-2023-0056789",
    },

    # Reports
    "report_maritime_trend": {
        "id": "report_maritime_trend", "type": "Report",
        "name": "2023 해양 스마트 기술 트렌드 보고서",
        "year": 2023,
        "summary": "자율운항, 친환경 추진, 디지털 트윈 등 해양 기술 2023년 동향 분석",
        "report_type": "기술동향",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # 도메인 2: AI/바이오 신약 개발 (AI Drug Discovery)
    # ══════════════════════════════════════════════════════════════════════════

    # Organizations
    "org_bio_kaist": {
        "id": "org_bio_kaist", "type": "Organization",
        "name": "KAIST 바이오및뇌공학과",
        "text": "AI 기반 신약 개발 및 단백질 구조 예측 연구",
    },
    "org_genexine": {
        "id": "org_genexine", "type": "Organization",
        "name": "제넥신",
        "text": "플랫폼 기술 기반 항암제·희귀질환 치료제 개발 바이오 제약사",
    },
    "org_standigm": {
        "id": "org_standigm", "type": "Organization",
        "name": "스탠다임",
        "text": "AI 신약 개발 플랫폼 Standigm BEST™ 운영 바이오테크 스타트업",
    },

    # Researchers
    "r_choi_bio": {
        "id": "r_choi_bio", "type": "Researcher",
        "name": "최바이오",
        "expertise": "딥러닝 단백질 구조 예측 분자 도킹 신약 후보 발굴",
        "topic": "AI 신약 바이오",
    },
    "r_jung_pharma": {
        "id": "r_jung_pharma", "type": "Researcher",
        "name": "정제약",
        "expertise": "ADMET 예측 약물 설계 생성 모델 GAN VAE",
        "topic": "신약 설계 생성AI",
    },
    "r_han_genomics": {
        "id": "r_han_genomics", "type": "Researcher",
        "name": "한유전체",
        "expertise": "유전체 분석 CRISPR 바이오마커 임상 데이터 AI",
        "topic": "유전체 바이오마커",
    },

    # Projects
    "proj_ai_drug": {
        "id": "proj_ai_drug", "type": "Project",
        "name": "AI 기반 항암제 신약 후보물질 발굴",
        "year": 2024,
        "text": "그래프 신경망과 트랜스포머 모델을 활용한 표적 단백질 결합 예측 및 신약 후보 스크리닝",
        "topic": "AI 신약 항암제 그래프신경망",
    },
    "proj_protein_fold": {
        "id": "proj_protein_fold", "type": "Project",
        "name": "단백질 3D 구조 예측 플랫폼 개발",
        "year": 2023,
        "text": "AlphaFold2 기반 파인튜닝 모델로 희귀 단백질 구조 예측 정확도 향상",
        "topic": "단백질 구조 AlphaFold",
    },

    # Papers
    "paper_gnn_drug": {
        "id": "paper_gnn_drug", "type": "Paper",
        "name": "그래프 신경망 기반 분자 독성 예측",
        "year": 2024,
        "abstract": "GNN과 어텐션 메커니즘을 결합한 분자 그래프 표현으로 ADMET 독성 예측 정확도 92% 달성.",
        "keywords": "GNN ADMET 독성 신약 분자",
    },
    "paper_alphafold_rare": {
        "id": "paper_alphafold_rare", "type": "Paper",
        "name": "희귀질환 표적 단백질 구조 예측 및 가상 스크리닝",
        "year": 2023,
        "abstract": "AlphaFold2 예측 구조에 분자 도킹을 적용하여 희귀질환 치료 후보물질 23종 발굴.",
        "keywords": "AlphaFold 가상스크리닝 희귀질환 단백질",
    },
    "paper_generative_mol": {
        "id": "paper_generative_mol", "type": "Paper",
        "name": "조건부 VAE를 이용한 신약 분자 생성",
        "year": 2024,
        "abstract": "표적 단백질 조건부 VAE로 선택성 높은 신규 약물 분자 구조를 자동 생성하는 방법론 제시.",
        "keywords": "VAE 생성모델 신약분자 약물설계",
    },

    # Patents
    "patent_ai_drug_screen": {
        "id": "patent_ai_drug_screen", "type": "Patent",
        "name": "AI 기반 신약 후보물질 스크리닝 시스템",
        "year": 2024,
        "summary": "다중 오믹스 데이터와 GNN을 결합한 신약 후보 물질 자동 스크리닝 특허",
        "patent_number": "KR-2024-0012345",
    },

    # Reports
    "report_ai_pharma": {
        "id": "report_ai_pharma", "type": "Report",
        "name": "AI 신약 개발 글로벌 동향 2024",
        "year": 2024,
        "summary": "생성 AI, 단백질 언어 모델, 임상 데이터 AI 분석 등 AI 신약 개발 2024년 글로벌 동향",
        "report_type": "기술동향",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # 도메인 3: 우주항공 (Aerospace / Small Satellites)
    # ══════════════════════════════════════════════════════════════════════════

    # Organizations
    "org_kari": {
        "id": "org_kari", "type": "Organization",
        "name": "한국항공우주연구원 KARI",
        "text": "위성 발사체 항공기 개발 국가 항공우주 연구기관",
    },
    "org_satrec": {
        "id": "org_satrec", "type": "Organization",
        "name": "쎄트렉아이",
        "text": "초소형·소형 지구관측위성 개발 및 데이터 서비스 전문 우주기업",
    },
    "org_innospace": {
        "id": "org_innospace", "type": "Organization",
        "name": "이노스페이스",
        "text": "소형 위성 발사용 하이브리드 로켓 한빛 시리즈 개발 스타트업",
    },

    # Researchers
    "r_oh_satellite": {
        "id": "r_oh_satellite", "type": "Researcher",
        "name": "오위성",
        "expertise": "초소형 위성 SAR 지구관측 위성 영상 처리",
        "topic": "위성 지구관측",
    },
    "r_shin_launch": {
        "id": "r_shin_launch", "type": "Researcher",
        "name": "신발사체",
        "expertise": "하이브리드 로켓 추진제 연소 궤도 최적화",
        "topic": "발사체 추진",
    },
    "r_yoon_space": {
        "id": "r_yoon_space", "type": "Researcher",
        "name": "윤우주",
        "expertise": "위성 군집 운용 우주 교통 관리 탑재체 설계",
        "topic": "위성 군집 운용",
    },

    # Projects
    "proj_cubesat": {
        "id": "proj_cubesat", "type": "Project",
        "name": "6U 큐브샛 SAR 위성 개발",
        "year": 2024,
        "text": "6U 큐브샛 플랫폼 기반 X밴드 SAR 탑재체 설계 및 소형 위성 검증",
        "topic": "큐브샛 SAR 소형위성",
    },
    "proj_launch_vehicle": {
        "id": "proj_launch_vehicle", "type": "Project",
        "name": "소형 발사체 한빛-TLV 상용화",
        "year": 2023,
        "text": "하이브리드 추진 방식의 소형 상용 발사체 설계 최적화 및 인증 비행 시험",
        "topic": "소형발사체 하이브리드로켓",
    },

    # Papers
    "paper_sar_cubesat": {
        "id": "paper_sar_cubesat", "type": "Paper",
        "name": "큐브샛 X밴드 SAR 안테나 설계 최적화",
        "year": 2024,
        "abstract": "6U 큐브샛 제약 조건에서 X밴드 SAR 패치 안테나 배열 설계. ISLR -18 dB 이하 달성.",
        "keywords": "SAR 큐브샛 안테나 위성",
    },
    "paper_hybrid_rocket": {
        "id": "paper_hybrid_rocket", "type": "Paper",
        "name": "HTPB/N2O 하이브리드 로켓 연소 특성 분석",
        "year": 2023,
        "abstract": "HTPB 고체 연료와 N2O 산화제 조합의 연소 효율 및 비추력 향상 방법 실험 연구.",
        "keywords": "하이브리드로켓 HTPB N2O 추진",
    },

    # Patents
    "patent_sat_payload": {
        "id": "patent_sat_payload", "type": "Patent",
        "name": "초소형 위성용 다중밴드 탑재체 제어 장치",
        "year": 2024,
        "summary": "저전력 고집적 초소형 위성 탑재체 통합 제어 및 열관리 특허",
        "patent_number": "KR-2024-0098765",
    },

    # Reports
    "report_space_trend": {
        "id": "report_space_trend", "type": "Report",
        "name": "뉴스페이스 산업 현황 및 전망 2024",
        "year": 2024,
        "summary": "초소형 위성 군집, 재사용 발사체, 우주 인터넷 등 뉴스페이스 산업 동향 분석",
        "report_type": "산업분석",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # 도메인 4: 수소/재생에너지 (Hydrogen & Renewable Energy)
    # ══════════════════════════════════════════════════════════════════════════

    # Organizations
    "org_kier": {
        "id": "org_kier", "type": "Organization",
        "name": "한국에너지기술연구원 KIER",
        "text": "수소 에너지 태양광 풍력 에너지 저장 전문 정부출연연구원",
    },
    "org_hyundai_hydrogen": {
        "id": "org_hyundai_hydrogen", "type": "Organization",
        "name": "현대차 수소연료전지사업부",
        "text": "수소연료전지 자동차 및 선박용 연료전지 시스템 개발",
    },
    "org_doosan_fuel": {
        "id": "org_doosan_fuel", "type": "Organization",
        "name": "두산퓨얼셀",
        "text": "인산형 연료전지 PAFC 및 고체산화물 연료전지 SOFC 발전 시스템 제조",
    },

    # Researchers
    "r_kwon_hydrogen": {
        "id": "r_kwon_hydrogen", "type": "Researcher",
        "name": "권수소",
        "expertise": "수전해 수소 생산 PEM 전해조 촉매 개발",
        "topic": "수소 수전해",
    },
    "r_lim_wind": {
        "id": "r_lim_wind", "type": "Researcher",
        "name": "임풍력",
        "expertise": "부유식 해상풍력 구조 설계 블레이드 공력 해석",
        "topic": "해상풍력 부유식",
    },
    "r_song_fuelcell": {
        "id": "r_song_fuelcell", "type": "Researcher",
        "name": "송연료전지",
        "expertise": "PEMFC 스택 열관리 내구성 향상 수소저장 합금",
        "topic": "연료전지 PEMFC",
    },

    # Projects
    "proj_green_hydrogen": {
        "id": "proj_green_hydrogen", "type": "Project",
        "name": "재생에너지 연계 그린수소 생산 실증",
        "year": 2024,
        "text": "해상풍력 잉여 전력을 활용한 PEM 수전해 그린수소 생산 및 저장 실증 플랜트",
        "topic": "그린수소 PEM 해상풍력 실증",
    },
    "proj_offshore_wind": {
        "id": "proj_offshore_wind", "type": "Project",
        "name": "1GW 부유식 해상풍력 단지 개발",
        "year": 2023,
        "text": "수심 100m 이상 심해 부유식 해상풍력 터빈 설계 및 계통 연계 기술 개발",
        "topic": "부유식 해상풍력 심해 계통",
    },

    # Papers
    "paper_pem_electrolyzer": {
        "id": "paper_pem_electrolyzer", "type": "Paper",
        "name": "고압 PEM 수전해조 막전극 접합체 최적화",
        "year": 2024,
        "abstract": "30 bar 고압 운전 PEM 수전해조 MEA 설계 최적화로 수소 생산 효율 82% 달성.",
        "keywords": "PEM 수전해 수소 MEA 고압",
    },
    "paper_floating_wind": {
        "id": "paper_floating_wind", "type": "Paper",
        "name": "부유식 해상풍력 계류 시스템 피로 수명 해석",
        "year": 2023,
        "abstract": "FOWT 계류 라인 동적 하중 스펙트럼 분석 및 피로 수명 예측 모델 개발.",
        "keywords": "부유식 해상풍력 FOWT 계류 피로",
    },
    "paper_h2_storage": {
        "id": "paper_h2_storage", "type": "Paper",
        "name": "금속 수소화물 기반 고밀도 수소 저장 연구",
        "year": 2024,
        "abstract": "Mg 기반 금속수소화물의 수소 저장 밀도 향상 및 흡·방출 속도 개선 연구.",
        "keywords": "수소저장 금속수소화물 Mg 에너지저장",
    },

    # Patents
    "patent_green_h2": {
        "id": "patent_green_h2", "type": "Patent",
        "name": "재생에너지 연계 수전해 수소 생산 제어 방법",
        "year": 2024,
        "summary": "간헐적 재생에너지 출력 변동에 대응하는 수전해 운전 제어 알고리즘 특허",
        "patent_number": "KR-2024-0034567",
    },
    "patent_fuel_cell_stack": {
        "id": "patent_fuel_cell_stack", "type": "Patent",
        "name": "고내구성 PEMFC 스택 열·수분 관리 장치",
        "year": 2023,
        "summary": "연료전지 스택 온도 균일화 및 수분 자동조절 내구성 향상 특허",
        "patent_number": "KR-2023-0078901",
    },

    # Reports
    "report_hydrogen_policy": {
        "id": "report_hydrogen_policy", "type": "Report",
        "name": "국내 수소 경제 로드맵 2030",
        "year": 2023,
        "summary": "그린수소 생산 목표, 수소 모빌리티 보급, 수소 발전 전환 정책 및 투자 계획",
        "report_type": "정책보고서",
    },
    "report_renewable_2024": {
        "id": "report_renewable_2024", "type": "Report",
        "name": "재생에너지 보급 현황 및 기술 전망 2024",
        "year": 2024,
        "summary": "태양광·해상풍력·수소 연계 재생에너지 국내외 보급 현황 및 2030년 기술 전망",
        "report_type": "기술동향",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# RELATIONS  {relation_type → [{from, to}]}
# ─────────────────────────────────────────────────────────────────────────────

SEED_RELATIONS: Dict[str, List[Dict[str, str]]] = {

    # ── AFFILIATED_WITH (Researcher → Organization) ────────────────────────
    "AFFILIATED_WITH": [
        # 해양
        {"from": "r_kim_ocean",     "to": "org_kaist_ocean"},
        {"from": "r_lee_maritime",  "to": "org_kriso"},
        {"from": "r_park_ocean",    "to": "org_samsung_heavy"},
        # AI/바이오
        {"from": "r_choi_bio",      "to": "org_bio_kaist"},
        {"from": "r_jung_pharma",   "to": "org_standigm"},
        {"from": "r_han_genomics",  "to": "org_genexine"},
        # 우주항공
        {"from": "r_oh_satellite",  "to": "org_satrec"},
        {"from": "r_shin_launch",   "to": "org_innospace"},
        {"from": "r_yoon_space",    "to": "org_kari"},
        # 수소/재생에너지
        {"from": "r_kwon_hydrogen", "to": "org_kier"},
        {"from": "r_lim_wind",      "to": "org_kier"},
        {"from": "r_song_fuelcell", "to": "org_hyundai_hydrogen"},
    ],

    # ── PARTICIPATED_IN (Researcher → Project) ─────────────────────────────
    "PARTICIPATED_IN": [
        # 해양
        {"from": "r_lee_maritime",  "to": "proj_autonomous_ship"},
        {"from": "r_kim_ocean",     "to": "proj_autonomous_ship"},
        {"from": "r_park_ocean",    "to": "proj_green_ship"},
        # AI/바이오
        {"from": "r_choi_bio",      "to": "proj_ai_drug"},
        {"from": "r_jung_pharma",   "to": "proj_ai_drug"},
        {"from": "r_choi_bio",      "to": "proj_protein_fold"},
        {"from": "r_han_genomics",  "to": "proj_protein_fold"},
        # 우주항공
        {"from": "r_oh_satellite",  "to": "proj_cubesat"},
        {"from": "r_yoon_space",    "to": "proj_cubesat"},
        {"from": "r_shin_launch",   "to": "proj_launch_vehicle"},
        # 수소/재생에너지
        {"from": "r_kwon_hydrogen", "to": "proj_green_hydrogen"},
        {"from": "r_lim_wind",      "to": "proj_offshore_wind"},
        {"from": "r_lim_wind",      "to": "proj_green_hydrogen"},
        {"from": "r_song_fuelcell", "to": "proj_green_hydrogen"},
    ],

    # ── AUTHORED (Researcher → Paper) ──────────────────────────────────────
    "AUTHORED": [
        # 해양
        {"from": "r_lee_maritime",  "to": "paper_ship_ai"},
        {"from": "r_kim_ocean",     "to": "paper_ship_ai"},
        {"from": "r_park_ocean",    "to": "paper_lng_propulsion"},
        # AI/바이오
        {"from": "r_choi_bio",      "to": "paper_gnn_drug"},
        {"from": "r_choi_bio",      "to": "paper_alphafold_rare"},
        {"from": "r_jung_pharma",   "to": "paper_generative_mol"},
        {"from": "r_han_genomics",  "to": "paper_alphafold_rare"},
        # 우주항공
        {"from": "r_oh_satellite",  "to": "paper_sar_cubesat"},
        {"from": "r_shin_launch",   "to": "paper_hybrid_rocket"},
        {"from": "r_yoon_space",    "to": "paper_sar_cubesat"},
        # 수소/재생에너지
        {"from": "r_kwon_hydrogen", "to": "paper_pem_electrolyzer"},
        {"from": "r_lim_wind",      "to": "paper_floating_wind"},
        {"from": "r_song_fuelcell", "to": "paper_h2_storage"},
        {"from": "r_kwon_hydrogen", "to": "paper_h2_storage"},
    ],

    # ── INVENTED (Researcher → Patent) ─────────────────────────────────────
    "INVENTED": [
        # 해양
        {"from": "r_lee_maritime",  "to": "patent_ship_nav"},
        # AI/바이오
        {"from": "r_choi_bio",      "to": "patent_ai_drug_screen"},
        {"from": "r_jung_pharma",   "to": "patent_ai_drug_screen"},
        # 우주항공
        {"from": "r_oh_satellite",  "to": "patent_sat_payload"},
        {"from": "r_yoon_space",    "to": "patent_sat_payload"},
        # 수소/재생에너지
        {"from": "r_kwon_hydrogen", "to": "patent_green_h2"},
        {"from": "r_lim_wind",      "to": "patent_green_h2"},
        {"from": "r_song_fuelcell", "to": "patent_fuel_cell_stack"},
    ],

    # ── PRODUCED (Organization → Project) ──────────────────────────────────
    "PRODUCED": [
        # 해양
        {"from": "org_kriso",           "to": "proj_autonomous_ship"},
        {"from": "org_samsung_heavy",   "to": "proj_green_ship"},
        # AI/바이오
        {"from": "org_bio_kaist",       "to": "proj_ai_drug"},
        {"from": "org_standigm",        "to": "proj_ai_drug"},
        {"from": "org_bio_kaist",       "to": "proj_protein_fold"},
        # 우주항공
        {"from": "org_satrec",          "to": "proj_cubesat"},
        {"from": "org_innospace",       "to": "proj_launch_vehicle"},
        {"from": "org_kari",            "to": "proj_launch_vehicle"},
        # 수소/재생에너지
        {"from": "org_kier",            "to": "proj_green_hydrogen"},
        {"from": "org_kier",            "to": "proj_offshore_wind"},
        {"from": "org_hyundai_hydrogen","to": "proj_green_hydrogen"},
    ],

    # ── CITES (Paper → Paper) ──────────────────────────────────────────────
    "CITES": [
        {"from": "paper_gnn_drug",        "to": "paper_alphafold_rare"},
        {"from": "paper_generative_mol",  "to": "paper_gnn_drug"},
        {"from": "paper_sar_cubesat",     "to": "paper_hybrid_rocket"},
        {"from": "paper_pem_electrolyzer","to": "paper_h2_storage"},
        {"from": "paper_floating_wind",   "to": "paper_pem_electrolyzer"},
    ],

    # ── PUBLISHED_IN (Project → Report) ────────────────────────────────────
    "PUBLISHED_IN": [
        {"from": "proj_autonomous_ship",  "to": "report_maritime_trend"},
        {"from": "proj_ai_drug",          "to": "report_ai_pharma"},
        {"from": "proj_cubesat",          "to": "report_space_trend"},
        {"from": "proj_green_hydrogen",   "to": "report_hydrogen_policy"},
        {"from": "proj_offshore_wind",    "to": "report_renewable_2024"},
    ],
}
