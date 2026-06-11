"""
Seed fixtures — SEED_NODES / SEED_RELATIONS
4개 도메인: 해양, AI/바이오 신약, 우주항공, 수소/재생에너지

필드 규칙:
  - Paper   : abstract (상세), keywords
  - Patent  : abstract (상세), keywords, patent_number
  - Report  : summary (상세), report_type
  - Project : text (검색용 요약), topic
  - Researcher: expertise, topic
  - Organization: text (검색용 설명)

Milvus에는 name+text(abstract/summary)+topic/expertise/keywords 조합이 들어가고,
MariaDB에는 abstract/summary 원문 전체 + keywords + authors(paper_authors 테이블)가 저장된다.
get_details_by_ids 호출 시 Milvus 기본 정보보다 훨씬 상세한 내용을 반환하는 것이 목표.
"""

from __future__ import annotations
from typing import Dict, List, Any

SEED_NODES: Dict[str, Dict[str, Any]] = {

    # ══════════════════════════════════════════════════════════════════════════
    # 도메인 1: 해양 (Maritime)
    # ══════════════════════════════════════════════════════════════════════════

    "org_kaist_ocean": {
        "id": "org_kaist_ocean", "type": "Organization",
        "name": "KAIST 해양시스템공학과",
        "text": "한국과학기술원 해양시스템 및 조선공학 전문 연구기관. "
                "선박 유체역학, 자율운항, 해양구조물 분야 국내 최고 수준의 연구 역량을 보유하며 "
                "산학협력 및 국가 R&D 과제를 다수 수행 중이다.",
    },
    "org_kriso": {
        "id": "org_kriso", "type": "Organization",
        "name": "한국해양과학기술원 KRISO",
        "text": "선박 및 해양구조물 연구개발 전문 정부출연연구원. "
                "자율운항선박 핵심기술, 친환경 추진 시스템, 해양 안전 기술 분야에서 "
                "연간 300억 원 이상의 R&D를 수행하는 해양 분야 대표 출연연이다.",
    },
    "org_samsung_heavy": {
        "id": "org_samsung_heavy", "type": "Organization",
        "name": "삼성중공업",
        "text": "초대형 컨테이너선, LNG선, 해양플랜트 건조 전문 조선사. "
                "자체 스마트십 플랫폼 SHI-MIND를 운영하며 디지털 트윈, "
                "자율운항 기술 내재화를 적극 추진 중이다.",
    },

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
    "r_chung_ship": {
        "id": "r_chung_ship", "type": "Researcher",
        "name": "정선박",
        "expertise": "소형 선박 보트 선형 설계 레저보트 유체역학 추진 성능",
        "topic": "보트 소형선박",
    },

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

    "paper_ship_ai": {
        "id": "paper_ship_ai", "type": "Paper",
        "name": "딥러닝 기반 선박 충돌 회피 알고리즘",
        "year": 2023,
        "abstract": (
            "본 논문은 강화학습(RL)과 LSTM 네트워크를 결합한 선박 자율운항 충돌 회피 시스템을 제안한다. "
            "COLREGs(국제 해상 충돌 예방 규칙) 준수를 위해 규칙 기반 제약 레이어를 RL 에이전트 위에 적층하였으며, "
            "시뮬레이션 환경에서 다중 선박 조우 시나리오 500건에 대해 충돌 회피 성공률 98.3%를 달성하였다. "
            "LSTM은 주변 선박의 AIS 기반 궤적을 10분 단위로 예측하여 미래 위협 판단에 활용된다. "
            "본 알고리즘은 실선 시험 결과 응답 지연 1.2초 이내로 국제 자율운항 4등급 요건을 충족함을 확인하였다."
        ),
        "keywords": "자율운항 충돌회피 강화학습 LSTM COLREGs 선박 AIS",
    },
    "paper_lng_propulsion": {
        "id": "paper_lng_propulsion", "type": "Paper",
        "name": "LNG 이중연료 엔진 성능 최적화 연구",
        "year": 2022,
        "abstract": (
            "본 연구는 LNG(액화천연가스)와 HFO(중유) 이중연료 2행정 저속 디젤 엔진의 연소 특성을 분석하고 "
            "NOx·SOx 배출 저감을 위한 연료 분사 타이밍 최적화 방법을 제시한다. "
            "MARPOL Annex VI Tier III NOx 기준 준수를 위해 EGR(배기가스 재순환)과 SCR 병행 적용 시 "
            "NOx 90% 저감이 가능함을 실험으로 검증하였다. "
            "연료 전환 과도 구간에서의 압력 맥동 제어 알고리즘을 새로 개발하여 엔진 안정성을 향상시켰으며, "
            "LNG 모드 운전 시 HFO 대비 CO2 25% 저감 효과를 확인하였다."
        ),
        "keywords": "LNG 이중연료 엔진 NOx SOx MARPOL 배출가스 저감",
    },
    "paper_boat_design": {
        "id": "paper_boat_design", "type": "Paper",
        "name": "레저 보트 선형 및 추진 성능 최적화 설계",
        "year": 2024,
        "abstract": (
            "CFD(전산유체역학) 시뮬레이션을 활용하여 8~12m급 레저 보트의 선형(hull form)을 최적화하고 "
            "전기 추진 시스템과의 통합 설계 방법론을 제안한다. "
            "NSGA-II 다목적 최적화 알고리즘으로 저항 최소화와 거주성 확보를 동시에 달성하는 파레토 최적 선형 집합을 도출하였다. "
            "최적 선형 적용 시 기존 선형 대비 전기 추진 에너지 소비량 18% 감소, "
            "최대 속력 12% 향상 결과를 수조 모형 실험을 통해 검증하였다. "
            "본 설계 방법론은 친환경 레저보트 인증 절차에 직접 적용 가능하다."
        ),
        "keywords": "보트 선형 레저보트 전기추진 CFD NSGA-II 최적설계 소형선박",
    },

    "patent_ship_nav": {
        "id": "patent_ship_nav", "type": "Patent",
        "name": "AI 기반 선박 경로 최적화 장치 및 방법",
        "year": 2023,
        "abstract": (
            "본 발명은 딥러닝 모델을 활용하여 기상·해상 조건, AIS 선박 밀도, 항만 스케줄을 실시간으로 분석하고 "
            "연료 소비 최소화 및 도착 시각 준수를 동시에 달성하는 선박 항로 최적화 시스템에 관한 것이다. "
            "청구항 1은 트랜스포머 기반 예측 모델을 이용한 기상 리스크 지수 산출 방법을 포함하며, "
            "청구항 2~5는 동적 프로그래밍 기반 실시간 항로 재계산 장치 구성을 청구한다."
        ),
        "keywords": "항로최적화 딥러닝 기상예측 AIS 연료절감 자율운항",
        "patent_number": "KR-2023-0056789",
    },

    "report_maritime_trend": {
        "id": "report_maritime_trend", "type": "Report",
        "name": "2023 해양 스마트 기술 트렌드 보고서",
        "year": 2023,
        "summary": (
            "본 보고서는 2023년 해양 분야 스마트 기술 동향을 자율운항, 친환경 추진, 디지털 트윈 세 축으로 분석한다. "
            "자율운항 분야에서는 IMO 자율운항 등급 4단계 로드맵과 국내 규제 샌드박스 현황을 정리하였으며, "
            "친환경 추진 분야에서는 LNG·메탄올·암모니아 연료 전환 비용 비교 분석을 수록하였다. "
            "디지털 트윈 부문에서는 삼성중공업 SHI-MIND, HD한국조선해양 HiMSEN 플랫폼 사례를 심층 분석하였다. "
            "2025년까지 자율운항 기술 상용화를 위한 국내 정책 과제와 R&D 투자 우선순위를 제언한다."
        ),
        "report_type": "기술동향",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # 도메인 2: AI/바이오 신약 개발
    # ══════════════════════════════════════════════════════════════════════════

    "org_bio_kaist": {
        "id": "org_bio_kaist", "type": "Organization",
        "name": "KAIST 바이오및뇌공학과",
        "text": "AI 기반 신약 개발 및 단백질 구조 예측 연구. "
                "그래프 신경망, 생성 모델, 멀티오믹스 통합 분석 분야에서 "
                "국제 수준의 연구 성과를 창출하고 있으며 글로벌 제약사와의 공동 연구를 활발히 수행 중이다.",
    },
    "org_genexine": {
        "id": "org_genexine", "type": "Organization",
        "name": "제넥신",
        "text": "플랫폼 기술 기반 항암제·희귀질환 치료제 개발 바이오 제약사. "
                "hyFc 융합 단백질 플랫폼 기술을 보유하며 GX-17(IL-7-hyFc) 등 "
                "다수의 파이프라인을 임상 2~3상에서 진행 중이다.",
    },
    "org_standigm": {
        "id": "org_standigm", "type": "Organization",
        "name": "스탠다임",
        "text": "AI 신약 개발 플랫폼 Standigm BEST™ 운영 바이오테크 스타트업. "
                "지식 그래프와 그래프 신경망을 결합한 표적 예측 플랫폼을 보유하며 "
                "글로벌 제약사와 5건 이상의 공동 연구를 진행 중이다.",
    },

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

    "paper_gnn_drug": {
        "id": "paper_gnn_drug", "type": "Paper",
        "name": "그래프 신경망 기반 분자 독성 예측",
        "year": 2024,
        "abstract": (
            "본 논문은 분자 구조를 그래프로 표현하고 멀티헤드 어텐션 기반 GNN(Graph Neural Network)으로 "
            "ADMET(흡수·분포·대사·배설·독성) 프로파일을 동시 예측하는 모델을 제안한다. "
            "Tox21, ClinTox, BBBP 등 10개 벤치마크 데이터셋에서 기존 최고 성능 대비 "
            "평균 AUC 3.2% 향상을 달성하였으며, 독성 예측 정확도 92%를 기록하였다. "
            "Edge 레벨 어텐션 시각화를 통해 독성 유발 작용기(functional group)를 해석 가능하게 설명하며, "
            "신약 후보 조기 탈락(early-stage attrition) 비용을 30% 절감할 수 있음을 시뮬레이션으로 확인하였다."
        ),
        "keywords": "GNN ADMET 독성예측 신약 분자 어텐션 벤치마크",
    },
    "paper_alphafold_rare": {
        "id": "paper_alphafold_rare", "type": "Paper",
        "name": "희귀질환 표적 단백질 구조 예측 및 가상 스크리닝",
        "year": 2023,
        "abstract": (
            "AlphaFold2로 예측한 구조적 데이터가 불충분한 희귀질환 단백질 12종에 대해 "
            "도메인 특화 파인튜닝 전략을 적용하여 예측 정확도를 TM-score 기준 0.12 향상시켰다. "
            "예측된 구조에 AutoDock-Vina 기반 분자 도킹을 적용하여 FDA 승인 약물 6,800종의 "
            "재목적화(drug repurposing) 가능성을 스크리닝하였으며, 후보물질 23종을 발굴하였다. "
            "발굴된 후보물질 중 5종은 in vitro 세포 실험에서 IC50 100 nM 이하의 활성을 확인하였으며, "
            "이 중 2종은 현재 전임상 단계에 진입하였다."
        ),
        "keywords": "AlphaFold 가상스크리닝 희귀질환 단백질 drug-repurposing 분자도킹",
    },
    "paper_generative_mol": {
        "id": "paper_generative_mol", "type": "Paper",
        "name": "조건부 VAE를 이용한 신약 분자 생성",
        "year": 2024,
        "abstract": (
            "표적 단백질의 결합 포켓 구조를 조건 벡터로 인코딩하여 선택적 결합이 가능한 "
            "신규 약물 분자를 생성하는 조건부 변분 오토인코더(CVAE) 프레임워크를 제안한다. "
            "생성된 분자의 유효성(validity) 97.8%, 신규성(novelty) 86.3%, 다양성(diversity) 0.73으로 "
            "기존 REINVENT, GCPN 대비 균형 잡힌 성능을 달성하였다. "
            "EGFR, CDK2, BRAF 3개 암 표적에 대해 생성된 분자 중 각각 41%, 38%, 45%가 "
            "기존 알려진 저해제와 유사한 결합 에너지를 보였으며, "
            "RDKit 기반 합성 가능성(SA score) 분석에서 생성 분자의 89%가 합성 용이 범위에 해당하였다."
        ),
        "keywords": "VAE 생성모델 신약분자 약물설계 CVAE EGFR 조건부생성",
    },

    "patent_ai_drug_screen": {
        "id": "patent_ai_drug_screen", "type": "Patent",
        "name": "AI 기반 신약 후보물질 스크리닝 시스템",
        "year": 2024,
        "abstract": (
            "본 발명은 유전체(genomics), 전사체(transcriptomics), 단백질체(proteomics) 등 "
            "다중 오믹스 데이터를 통합 처리하는 GNN 기반 신약 후보물질 자동 스크리닝 시스템에 관한 것이다. "
            "청구항 1은 이종 그래프(heterogeneous graph) 위에서 질병-단백질-약물 삼중 관계를 학습하는 "
            "링크 예측 모듈을 포함한다. 청구항 2~4는 설명 가능 AI(XAI) 기반 후보물질 우선순위 산출 방법을, "
            "청구항 5는 의약품 부작용 자동 플래그 기능을 청구한다."
        ),
        "keywords": "신약스크리닝 다중오믹스 GNN 이종그래프 XAI 부작용예측",
        "patent_number": "KR-2024-0012345",
    },

    "report_ai_pharma": {
        "id": "report_ai_pharma", "type": "Report",
        "name": "AI 신약 개발 글로벌 동향 2024",
        "year": 2024,
        "summary": (
            "본 보고서는 2024년 글로벌 AI 신약 개발 동향을 생성 AI, 단백질 언어 모델(PLM), "
            "멀티오믹스 통합 분석, 임상 데이터 AI 네 분야로 분류하여 심층 분석한다. "
            "생성 AI 부문에서는 Insilico Medicine, Exscientia, Recursion 등 선도 기업의 "
            "임상 진입 파이프라인 현황을 정리하였다. "
            "단백질 언어 모델 부문에서는 ESM-2, ProGen2, ProteinMPNN의 신약 개발 적용 사례를 비교하였으며, "
            "국내 바이오벤처의 글로벌 기술 수준 격차 분석과 정책 지원 방향을 제언한다."
        ),
        "report_type": "기술동향",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # 도메인 3: 우주항공
    # ══════════════════════════════════════════════════════════════════════════

    "org_kari": {
        "id": "org_kari", "type": "Organization",
        "name": "한국항공우주연구원 KARI",
        "text": "위성, 발사체, 항공기 개발 국가 항공우주 연구기관. "
                "누리호(KSLV-Ⅱ) 발사 성공을 통해 독자 발사 역량을 확보하였으며 "
                "차세대 중형위성, 달 탐사선 다누리 개발을 주도하고 있다.",
    },
    "org_satrec": {
        "id": "org_satrec", "type": "Organization",
        "name": "쎄트렉아이",
        "text": "초소형·소형 지구관측위성 개발 및 데이터 서비스 전문 우주기업. "
                "자체 개발 위성 플랫폼 SI-300·SI-600을 기반으로 "
                "30개국 이상에 위성 및 지상국 시스템을 수출한 국내 대표 위성 전문기업이다.",
    },
    "org_innospace": {
        "id": "org_innospace", "type": "Organization",
        "name": "이노스페이스",
        "text": "소형 위성 발사용 하이브리드 로켓 한빛 시리즈 개발 스타트업. "
                "2023년 한빛-TLV 시험 발사에 성공하여 국내 민간 최초 우주 발사체 발사 성공 기록을 수립하였다.",
    },

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

    "paper_sar_cubesat": {
        "id": "paper_sar_cubesat", "type": "Paper",
        "name": "큐브샛 X밴드 SAR 안테나 설계 최적화",
        "year": 2024,
        "abstract": (
            "6U 큐브샛(30×20×10 cm, 8 kg 이하)의 공간·전력 제약 조건에서 "
            "X밴드(9.6 GHz) SAR 임무를 수행할 수 있는 능동 위상 배열 패치 안테나 설계 방법을 제안한다. "
            "16×4 패치 배열에 디지털 빔포밍을 적용하여 방위각 해상도 3 m, 고도각 해상도 5 m를 달성하였으며 "
            "ISLR(통합 부엽비) -18.4 dB 이하를 만족하였다. "
            "전력 소비 12 W, 질량 1.4 kg의 경량 설계로 6U 표준 슬롯 2개에 수납 가능함을 PDR 수준에서 검증하였다. "
            "본 설계는 저궤도 초소형 SAR 군집 위성 기반의 재방문 주기 30분 이하 지구관측 임무에 적용 가능하다."
        ),
        "keywords": "SAR 큐브샛 X밴드 패치안테나 위상배열 빔포밍 소형위성",
    },
    "paper_hybrid_rocket": {
        "id": "paper_hybrid_rocket", "type": "Paper",
        "name": "HTPB/N2O 하이브리드 로켓 연소 특성 분석",
        "year": 2023,
        "abstract": (
            "HTPB(폴리부타디엔) 고체 연료와 N2O(아산화질소) 기체 산화제를 사용하는 "
            "하이브리드 로켓 모터의 연소 특성을 직경 150 mm 스케일 시험체로 분석하였다. "
            "산화제 대 연료 비율(O/F ratio) 6.0~9.5 범위에서 연소 효율(c* 효율)이 최대 94.2%임을 확인하였으며, "
            "회귀율(regression rate) 모델을 실험 데이터로 보정하여 설계 정확도를 향상시켰다. "
            "비추력(Isp) 향상을 위해 알루미늄 분말 6 wt% 첨가 시 진공 Isp 268 s로 "
            "기준 대비 12 s 향상되는 결과를 얻었다. "
            "본 결과는 300 kg급 소형 발사체 1단 모터 설계에 직접 적용되었다."
        ),
        "keywords": "하이브리드로켓 HTPB N2O 연소효율 비추력 회귀율 소형발사체",
    },

    "patent_sat_payload": {
        "id": "patent_sat_payload", "type": "Patent",
        "name": "초소형 위성용 다중밴드 탑재체 제어 장치",
        "year": 2024,
        "abstract": (
            "본 발명은 3U 이하 초소형 위성에서 광학, SAR, AIS 수신기 등 다중 탑재체를 "
            "단일 OBC(탑재 컴퓨터)로 통합 제어하는 장치 및 방법에 관한 것이다. "
            "청구항 1은 탑재체별 전력 우선순위 동적 할당 알고리즘을 포함하며, "
            "청구항 2~3은 열진공 환경에서 탑재체 온도를 -20°C~+60°C 범위로 유지하는 "
            "수동형 열제어 구조를 청구한다. 청구항 4는 CCSDS 표준 기반 탑재체 데이터 패킷화 및 "
            "다운링크 우선순위 스케줄링 방법을 포함한다."
        ),
        "keywords": "초소형위성 탑재체제어 OBC 열관리 전력관리 CCSDS 다중밴드",
        "patent_number": "KR-2024-0098765",
    },

    "report_space_trend": {
        "id": "report_space_trend", "type": "Report",
        "name": "뉴스페이스 산업 현황 및 전망 2024",
        "year": 2024,
        "summary": (
            "본 보고서는 초소형 위성 군집(mega-constellation), 재사용 발사체, 우주 인터넷, "
            "달·화성 탐사 등 뉴스페이스 산업 4대 분야의 2024년 현황과 2030년 전망을 분석한다. "
            "SpaceX Starlink 6,000기 돌파, OneWeb·Amazon Kuiper 상용 서비스 개시 현황을 정리하였으며, "
            "국내 위성 군집 사업 추진 동향과 발사 서비스 자립화 로드맵을 별도 챕터로 수록하였다. "
            "뉴스페이스 시장 규모가 2030년 1조 달러 초과를 전망하는 분석 기관 5개사의 시나리오를 비교하고 "
            "국내 기업의 글로벌 가치사슬 편입 전략을 제언한다."
        ),
        "report_type": "산업분석",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # 도메인 4: 수소/재생에너지
    # ══════════════════════════════════════════════════════════════════════════

    "org_kier": {
        "id": "org_kier", "type": "Organization",
        "name": "한국에너지기술연구원 KIER",
        "text": "수소 에너지, 태양광, 풍력, 에너지 저장 전문 정부출연연구원. "
                "그린수소 생산 원가 절감, 해상풍력 대형화, 차세대 배터리 등 "
                "탄소중립 핵심 기술 개발을 주도하며 연간 2,000억 원 이상의 R&D를 수행한다.",
    },
    "org_hyundai_hydrogen": {
        "id": "org_hyundai_hydrogen", "type": "Organization",
        "name": "현대차 수소연료전지사업부",
        "text": "수소연료전지 자동차 및 선박용 연료전지 시스템 개발. "
                "NEXO 탑재 100 kW급 수소연료전지 시스템과 "
                "선박용 1 MW급 PEMFC 파워팩을 개발하여 상용화 단계에 있다.",
    },
    "org_doosan_fuel": {
        "id": "org_doosan_fuel", "type": "Organization",
        "name": "두산퓨얼셀",
        "text": "인산형 연료전지 PAFC 및 고체산화물 연료전지 SOFC 발전 시스템 제조. "
                "440 kW PAFC 시스템을 국내외 수소 발전 시장에 공급하며 "
                "SOFC 기반 건물용 분산 발전 시스템을 개발 중이다.",
    },

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

    "paper_pem_electrolyzer": {
        "id": "paper_pem_electrolyzer", "type": "Paper",
        "name": "고압 PEM 수전해조 막전극 접합체 최적화",
        "year": 2024,
        "abstract": (
            "30 bar 고압 운전 조건에서 PEM(양성자교환막) 수전해조 성능을 극대화하기 위한 "
            "막전극 접합체(MEA) 설계 최적화 방법을 제시한다. "
            "Pt 로딩량 0.3 mg/cm² 이하의 저백금 애노드 촉매층을 iridium oxide와 복합화하여 "
            "과전압 30 mV 감소를 달성하였으며, 불소화 이오노머 농도 최적화로 "
            "막 저항을 15% 저감하였다. "
            "70°C, 30 bar 조건에서 전류밀도 3 A/cm²에서 수소 생산 효율 82%를 기록하였으며, "
            "1,000시간 내구성 시험에서 성능 감소율 2% 이하를 유지하였다. "
            "본 MEA 설계는 재생에너지 연계 MW급 수전해 시스템에 직접 적용 가능하다."
        ),
        "keywords": "PEM 수전해 수소 MEA 고압 저백금 촉매 내구성",
    },
    "paper_floating_wind": {
        "id": "paper_floating_wind", "type": "Paper",
        "name": "부유식 해상풍력 계류 시스템 피로 수명 해석",
        "year": 2023,
        "abstract": (
            "수심 120~200 m 해역에 설치되는 15 MW급 부유식 해상풍력 발전기(FOWT)의 "
            "체인-와이어 복합 계류 라인에 대한 피로 수명 해석 방법론을 제안한다. "
            "ORCAFLEX 기반 시간 영역 해석으로 10년치 파랑·바람·조류 조건을 통계 처리한 "
            "하중 스펙트럼을 구성하였으며, DNV GL S-N 곡선을 적용하여 계류 라인 피로 수명을 "
            "안전계수 10 조건에서 25년 이상으로 확인하였다. "
            "핫스팟 부위는 페어리더 연결부임을 확인하였으며, 연결부 형상 최적화로 "
            "응력 집중 계수(SCF)를 18% 저감하는 설계 개선안을 제시하였다."
        ),
        "keywords": "부유식 해상풍력 FOWT 계류 피로수명 ORCAFLEX S-N곡선 심해",
    },
    "paper_h2_storage": {
        "id": "paper_h2_storage", "type": "Paper",
        "name": "금속 수소화물 기반 고밀도 수소 저장 연구",
        "year": 2024,
        "abstract": (
            "Mg₂FeH₆ 및 Mg₂NiH₄ 복합 금속수소화물 소재의 수소 저장 밀도 향상 및 "
            "흡·방출 속도 개선을 위한 나노 복합화 전략을 연구하였다. "
            "볼밀링과 스퍼터링을 결합한 나노 복합화 공정으로 평균 입자 크기를 80 nm로 제어하여 "
            "수소화 반응 속도를 기존 마이크로 입자 대비 4.2배 향상시켰다. "
            "저장 밀도는 6.1 wt%로 DOE 2025 목표치(5.5 wt%)를 상회하였으며, "
            "300°C 이하 방출 조건에서도 완전 방출이 가능함을 TGA 분석으로 확인하였다. "
            "본 소재는 고정형 수소 저장 스테이션과 선박용 수소 연료 시스템에의 적용 가능성을 제시한다."
        ),
        "keywords": "수소저장 금속수소화물 Mg 나노복합화 저장밀도 에너지저장 DOE",
    },

    "patent_green_h2": {
        "id": "patent_green_h2", "type": "Patent",
        "name": "재생에너지 연계 수전해 수소 생산 제어 방법",
        "year": 2024,
        "abstract": (
            "본 발명은 태양광·풍력의 간헐적 출력 변동에 실시간으로 대응하는 "
            "PEM 수전해 시스템 운전 제어 방법 및 장치에 관한 것이다. "
            "청구항 1은 5분 단위 재생에너지 출력 예측값을 입력으로 받아 "
            "수전해 스택 전류 밀도를 0.1 A/cm² 단위로 제어하는 MPC(모델예측제어) 알고리즘을 포함한다. "
            "청구항 2는 스택 과부하 방지를 위한 동적 안전 운전 범위 자동 설정 방법을, "
            "청구항 3~4는 수소 버퍼 탱크와 연계한 생산·저장 통합 스케줄링 방법을 청구한다."
        ),
        "keywords": "수전해 재생에너지 MPC 제어 수소생산 PEM 간헐성 버퍼탱크",
        "patent_number": "KR-2024-0034567",
    },
    "patent_fuel_cell_stack": {
        "id": "patent_fuel_cell_stack", "type": "Patent",
        "name": "고내구성 PEMFC 스택 열·수분 관리 장치",
        "year": 2023,
        "abstract": (
            "본 발명은 차량 및 선박용 100 kW급 PEMFC 스택에서 발생하는 열을 균일하게 제거하고 "
            "막 수분 상태를 최적으로 유지하는 통합 열·수분 관리 장치에 관한 것이다. "
            "청구항 1은 냉각수 유로 3D 최적화 설계로 스택 내 온도 편차를 ±2°C 이내로 제어하는 방법을, "
            "청구항 2는 전기화학 임피던스 스펙트로스코피(EIS) 기반 실시간 막 수분 진단 알고리즘을 청구한다. "
            "청구항 3은 가속 스트레스 시험(AST) 프로토콜로 5,000 시간 내구성을 확인한 스택 구조를 포함한다."
        ),
        "keywords": "PEMFC 열관리 수분관리 내구성 EIS 냉각수 스택 수소연료전지",
        "patent_number": "KR-2023-0078901",
    },

    "report_hydrogen_policy": {
        "id": "report_hydrogen_policy", "type": "Report",
        "name": "국내 수소 경제 로드맵 2030",
        "year": 2023,
        "summary": (
            "본 보고서는 정부의 수소경제 활성화 로드맵을 바탕으로 2030년까지의 "
            "그린수소 생산 목표(연 100만 톤), 수소 모빌리티 보급 계획(수소차 30만 대, "
            "수소버스 4만 대, 수소트럭 3만 대), 수소 발전 전환 정책(수소·암모니아 혼소 15 GW)을 분석한다. "
            "그린수소 생산 원가를 2030년 3,500원/kg 이하로 달성하기 위한 "
            "수전해 기술 개발, 해외 그린수소 도입, 인프라 구축 투자 계획을 수록하였다. "
            "규제 혁신, 세제 지원, R&D 예산 배분 등 정책 과제별 이행 현황 및 개선 방향을 제언한다."
        ),
        "report_type": "정책보고서",
    },
    "report_renewable_2024": {
        "id": "report_renewable_2024", "type": "Report",
        "name": "재생에너지 보급 현황 및 기술 전망 2024",
        "year": 2024,
        "summary": (
            "본 보고서는 2024년 국내외 재생에너지 보급 현황을 태양광, 해상풍력, 수소 연계 저장 세 부문으로 정리하고 "
            "2030년 기술 전망을 제시한다. "
            "국내 재생에너지 발전 비중이 처음으로 10%를 초과하였으며 "
            "해상풍력은 전북·전남·제주 해역에서 8.2 GW의 개발 허가가 완료되었다. "
            "기술 전망 부문에서는 페로브스카이트 태양전지(효율 28% 돌파), "
            "20 MW급 대형 해상풍력 터빈, 장주기 에너지 저장(LDES) 기술의 "
            "상용화 일정과 비용 목표를 국제 기관별로 비교 분석하였다."
        ),
        "report_type": "기술동향",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# RELATIONS
# ─────────────────────────────────────────────────────────────────────────────

SEED_RELATIONS: Dict[str, List[Dict[str, str]]] = {

    "AFFILIATED_WITH": [
        {"from": "r_kim_ocean",     "to": "org_kaist_ocean"},
        {"from": "r_lee_maritime",  "to": "org_kriso"},
        {"from": "r_chung_ship",    "to": "org_kriso"},
        {"from": "r_park_ocean",    "to": "org_samsung_heavy"},
        {"from": "r_choi_bio",      "to": "org_bio_kaist"},
        {"from": "r_jung_pharma",   "to": "org_standigm"},
        {"from": "r_han_genomics",  "to": "org_genexine"},
        {"from": "r_oh_satellite",  "to": "org_satrec"},
        {"from": "r_shin_launch",   "to": "org_innospace"},
        {"from": "r_yoon_space",    "to": "org_kari"},
        {"from": "r_kwon_hydrogen", "to": "org_kier"},
        {"from": "r_lim_wind",      "to": "org_kier"},
        {"from": "r_song_fuelcell", "to": "org_hyundai_hydrogen"},
    ],

    "PARTICIPATED_IN": [
        {"from": "r_lee_maritime",  "to": "proj_autonomous_ship"},
        {"from": "r_kim_ocean",     "to": "proj_autonomous_ship"},
        {"from": "r_park_ocean",    "to": "proj_green_ship"},
        {"from": "r_choi_bio",      "to": "proj_ai_drug"},
        {"from": "r_jung_pharma",   "to": "proj_ai_drug"},
        {"from": "r_choi_bio",      "to": "proj_protein_fold"},
        {"from": "r_han_genomics",  "to": "proj_protein_fold"},
        {"from": "r_oh_satellite",  "to": "proj_cubesat"},
        {"from": "r_yoon_space",    "to": "proj_cubesat"},
        {"from": "r_shin_launch",   "to": "proj_launch_vehicle"},
        {"from": "r_kwon_hydrogen", "to": "proj_green_hydrogen"},
        {"from": "r_lim_wind",      "to": "proj_offshore_wind"},
        {"from": "r_lim_wind",      "to": "proj_green_hydrogen"},
        {"from": "r_song_fuelcell", "to": "proj_green_hydrogen"},
    ],

    "AUTHORED": [
        {"from": "r_lee_maritime",  "to": "paper_ship_ai"},
        {"from": "r_kim_ocean",     "to": "paper_ship_ai"},
        {"from": "r_park_ocean",    "to": "paper_lng_propulsion"},
        {"from": "r_chung_ship",    "to": "paper_boat_design"},
        {"from": "r_choi_bio",      "to": "paper_gnn_drug"},
        {"from": "r_choi_bio",      "to": "paper_alphafold_rare"},
        {"from": "r_jung_pharma",   "to": "paper_generative_mol"},
        {"from": "r_han_genomics",  "to": "paper_alphafold_rare"},
        {"from": "r_oh_satellite",  "to": "paper_sar_cubesat"},
        {"from": "r_shin_launch",   "to": "paper_hybrid_rocket"},
        {"from": "r_yoon_space",    "to": "paper_sar_cubesat"},
        {"from": "r_kwon_hydrogen", "to": "paper_pem_electrolyzer"},
        {"from": "r_lim_wind",      "to": "paper_floating_wind"},
        {"from": "r_song_fuelcell", "to": "paper_h2_storage"},
        {"from": "r_kwon_hydrogen", "to": "paper_h2_storage"},
    ],

    "INVENTED": [
        {"from": "r_lee_maritime",  "to": "patent_ship_nav"},
        {"from": "r_choi_bio",      "to": "patent_ai_drug_screen"},
        {"from": "r_jung_pharma",   "to": "patent_ai_drug_screen"},
        {"from": "r_oh_satellite",  "to": "patent_sat_payload"},
        {"from": "r_yoon_space",    "to": "patent_sat_payload"},
        {"from": "r_kwon_hydrogen", "to": "patent_green_h2"},
        {"from": "r_lim_wind",      "to": "patent_green_h2"},
        {"from": "r_song_fuelcell", "to": "patent_fuel_cell_stack"},
    ],

    "PRODUCED": [
        {"from": "org_kriso",            "to": "proj_autonomous_ship"},
        {"from": "org_samsung_heavy",    "to": "proj_green_ship"},
        {"from": "org_bio_kaist",        "to": "proj_ai_drug"},
        {"from": "org_standigm",         "to": "proj_ai_drug"},
        {"from": "org_bio_kaist",        "to": "proj_protein_fold"},
        {"from": "org_satrec",           "to": "proj_cubesat"},
        {"from": "org_innospace",        "to": "proj_launch_vehicle"},
        {"from": "org_kari",             "to": "proj_launch_vehicle"},
        {"from": "org_kier",             "to": "proj_green_hydrogen"},
        {"from": "org_kier",             "to": "proj_offshore_wind"},
        {"from": "org_hyundai_hydrogen", "to": "proj_green_hydrogen"},
    ],

    "FILED": [
        {"from": "org_kriso",            "to": "patent_ship_nav"},
        {"from": "org_bio_kaist",        "to": "patent_ai_drug_screen"},
        {"from": "org_standigm",         "to": "patent_ai_drug_screen"},
        {"from": "org_satrec",           "to": "patent_sat_payload"},
        {"from": "org_kari",             "to": "patent_sat_payload"},
        {"from": "org_kier",             "to": "patent_green_h2"},
        {"from": "org_hyundai_hydrogen", "to": "patent_fuel_cell_stack"},
    ],

    "CITES": [
        {"from": "paper_gnn_drug",         "to": "paper_alphafold_rare"},
        {"from": "paper_generative_mol",   "to": "paper_gnn_drug"},
        {"from": "paper_sar_cubesat",      "to": "paper_hybrid_rocket"},
        {"from": "paper_pem_electrolyzer", "to": "paper_h2_storage"},
        {"from": "paper_floating_wind",    "to": "paper_pem_electrolyzer"},
    ],

    "PUBLISHED_IN": [
        {"from": "proj_autonomous_ship", "to": "report_maritime_trend"},
        {"from": "proj_ai_drug",         "to": "report_ai_pharma"},
        {"from": "proj_cubesat",         "to": "report_space_trend"},
        {"from": "proj_green_hydrogen",  "to": "report_hydrogen_policy"},
        {"from": "proj_offshore_wind",   "to": "report_renewable_2024"},
    ],
}
