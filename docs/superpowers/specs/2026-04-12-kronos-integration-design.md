# Kronos Foundation Model 통합 설계

> daily-stock-report + stock-analyzer 스킬에 Kronos-mini 확률적 예측을 통합한다.

## 배경

### 현재 시스템

- **daily-stock-report**: GitHub Actions (매일 2회) → yfinance 기반 규칙적 TA (RSI/MACD/BB) → HTML 이메일 리포트 (57종목)
- **stock-analyzer 스킬**: Claude Code 대화형 개별 종목 분석 (main.py 자체 완결형)
- 둘 다 규칙 기반 시그널만 제공. 확률적 예측, 변동성 예측 없음.

### Kronos

- 120억 캔들스틱으로 사전학습된 금융 Foundation Model (AAAI 2026)
- OHLCV 입력 → 토크나이저 → Autoregressive Transformer → 확률적 가격 경로 예측
- Kronos-mini (4.1M params, context 2048): CPU 추론 가능
- GitHub: https://github.com/shiyu-coder/Kronos
- HuggingFace: NeoQuasar/Kronos-mini

### 목표

기존 규칙 기반 TA 위에 "확률 분포 기반 가격/변동성 예측" 레이어를 추가하여 리포트와 스킬 양쪽의 분석 품질을 향상.

## 아키텍처

### 파일 구조

```
daily-stock-report/
├── libs/
│   └── kronos_predictor.py       ← NEW: Kronos 래퍼 (단일 소스)
├── scripts/
│   ├── ai_forecast.py            ← NEW: 배치 예측 → HTML [AI 예측] 섹션
│   ├── sort_utils.py             ← NEW: 시장/시총 기반 정렬 유틸리티
│   ├── report_generator.py       ← MODIFY: ai_forecast 호출 + 정렬 적용
│   ├── analyzer.py               (기존 유지, 정렬된 입력 수신)
│   ├── market_overview.py        (기존 유지)
│   ├── extras.py                 (기존 유지, 정렬된 입력 수신)
│   ├── gem_scanner.py            (기존 유지)
│   └── send_email.py             (기존 유지)
├── requirements.txt              ← MODIFY: torch, kronos 의존성 추가
└── .github/workflows/
    └── daily-report.yml          ← MODIFY: 타임아웃 20분

~/.claude/skills/stock-analyzer/
├── scripts/
│   └── main.py                   ← MODIFY: Kronos 예측 통합
└── SKILL.md                      ← MODIFY: forecast 출력 포맷 가이드 추가
```

### 데이터 흐름

```
yfinance OHLCV (기존)
    │
    ├─→ analyzer.py (RSI/MACD/BB — 기존 TA)
    │
    └─→ kronos_predictor.py
            │
            ├─ Kronos-mini 모델 로드 (HuggingFace, 최초 1회 → 디스크 캐시)
            ├─ OHLCV → 토크나이저 → 토큰 시퀀스
            ├─ Autoregressive 예측 (1일 + 5일 horizon)
            ├─ Monte Carlo 30경로 샘플링
            └─ 반환: {median, p10, p90, direction_prob, volatility}
                │
                ├─→ ai_forecast.py → HTML 섹션 (daily-stock-report)
                └─→ stock-analyzer main.py → JSON 출력 (스킬)
```

## 컴포넌트 상세

### 1. kronos_predictor.py (libs/)

Kronos 추론을 감싸는 단일 래퍼. daily-stock-report와 stock-analyzer 양쪽에서 사용.

```python
class KronosPredictor:
    def __init__(self, model_name="NeoQuasar/Kronos-mini", device="cpu"):
        """HuggingFace에서 모델+토크나이저 로드. 최초 1회, 이후 디스크 캐시."""

    def predict(self, df, horizons=[1, 5], n_samples=30):
        """
        단일 종목 예측.

        Parameters:
            df: pandas DataFrame (yfinance 형식 — Open, High, Low, Close, Volume)
                최소 60일 이상 일봉 데이터 권장
            horizons: 예측 기간 리스트 (거래일 단위)
            n_samples: Monte Carlo 샘플 수

        Returns: {
            "1d": {
                "median": 59200,
                "p10": 58100,
                "p90": 60300,
                "direction_prob": 0.62,
                "volatility": 0.018
            },
            "5d": { ... 동일 구조 ... }
        }
        """

    def predict_batch(self, tickers_data, horizons=[1, 5], n_samples=30,
                      timeout_seconds=600, fallback_tickers=None):
        """
        다종목 배치 예측 (타임아웃 지원).

        Parameters:
            tickers_data: {ticker: df} 딕셔너리
            timeout_seconds: 전체 타임아웃 (기본 10분)
            fallback_tickers: 타임아웃 시 우선 처리할 종목 set

        Returns: {ticker: predict() 결과 또는 None (스킵)}

        타임아웃 로직:
            Phase 1: fallback_tickers 15종목 우선 처리
            Phase 2: 나머지 종목 순차 처리
            경과 > timeout_seconds × 0.7 → 남은 종목 스킵, None 반환
        """
```

### 2. ai_forecast.py (scripts/)

배치 예측 실행 → HTML `[AI 예측]` 섹션 생성.

```python
FALLBACK_TICKERS = {
    # KOSPI 10 (시총 상위)
    "005930.KS",   # 삼성전자
    "000660.KS",   # SK하이닉스
    "005380.KS",   # 현대차
    "051910.KS",   # LG화학
    "373220.KS",   # LG에너지솔루션
    "006400.KS",   # 삼성SDI
    "035420.KS",   # 네이버
    "035720.KS",   # 카카오
    "017670.KS",   # SK텔레콤
    "259960.KS",   # 크래프톤
    # NASDAQ/NYSE 5 (시총 상위)
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "TSLA",
}

def generate_ai_forecast_section(watchlist, ticker_info_cache, timeout_seconds=600):
    """
    1. watchlist 전종목 OHLCV 수집 (yfinance, 90일)
    2. KronosPredictor.predict_batch() 호출
    3. HTML [AI 예측] 섹션 반환
    """
```

**HTML 출력 구조:**

```
┌──────────────────────────────────────────────────┐
│  📊 AI 예측 (Kronos-mini)                         │
├──────────────────────────────────────────────────┤
│                                                    │
│  ▸ 1일 예측                                        │
│  ┌────────┬────────┬──────────┬──────┬──────────┐ │
│  │ 종목   │ 현재가  │ 예상 범위 │ 방향  │ 상승확률  │ │
│  ├────────┼────────┼──────────┼──────┼──────────┤ │
│  │ 삼성전자│ 58,500 │58.1K~60.3K│ 🔼  │ 62%     │ │
│  │ SK하이닉스│ ...  │          │      │          │ │
│  │ ...    │        │          │      │          │ │
│  │ (KOSPI/KOSDAQ 시총 내림차순)                     │ │
│  │ NVDA   │ $138.2 │$135~$142 │ 🔼  │ 58%     │ │
│  │ AAPL   │ ...    │          │      │          │ │
│  │ ...    │        │          │      │          │ │
│  │ (US 시총 내림차순)                               │ │
│  └────────┴────────┴──────────┴──────┴──────────┘ │
│                                                    │
│  ▸ 5일(주간) 예측                                   │
│  ┌────────┬────────┬──────────┬──────┬──────────┐ │
│  │ 종목   │ 현재가  │ 예상 범위 │ 방향  │ 변동성   │ │
│  │ (동일 정렬)                                     │ │
│  └────────┴────────┴──────────┴──────┴──────────┘ │
│                                                    │
│  ⚠ AI 예측은 참고용이며 투자 권유가 아닙니다.         │
│  ⏱ 57종목 중 52종목 예측 완료 (5종목 타임아웃 스킵)   │
└──────────────────────────────────────────────────┘
```

**방향 아이콘:**
- 🔼 상승확률 ≥ 55%
- 🔽 상승확률 ≤ 45%
- ➡️ 그 사이 (횡보)

**변동성 레이블 (5일 예측):**
- 일간 volatility > 0.03: 높음
- 0.015 ~ 0.03: 보통
- < 0.015: 낮음

**스킵 종목:** 테이블에 "⏱ 스킵" 표시, 하단에 총 처리/스킵 카운트.

### 3. sort_utils.py (scripts/)

전체 리포트에서 사용하는 공통 정렬 유틸리티.

```python
def sort_by_market_and_cap(tickers, ticker_info_cache=None):
    """
    Parameters:
        tickers: [{"ticker": "005930.KS", "name": "삼성전자"}, ...]
        ticker_info_cache: {ticker: {"marketCap": int}}
                           None이면 yfinance stock.info에서 조회

    Returns: 정렬된 리스트

    정렬 기준:
        1차: 시장 그룹 — KRX (.KS, .KQ) 먼저, US 뒤
        2차: 각 그룹 내 marketCap 내림차순
        3차: marketCap 조회 실패 → 그룹 내 맨 뒤
    """
```

**적용 지점:** `report_generator.py`에서 워치리스트 로드 직후 1회 정렬 → 정렬된 리스트를 analyzer.py, extras.py, ai_forecast.py 모두에 전달.

**marketCap 캐시:** 57종목 `stock.info["marketCap"]` 조회 (~1분). `report_generator.py`에서 한 번 조회 후 딕셔너리로 각 모듈에 전달하여 중복 호출 방지.

### 4. report_generator.py 수정

```python
# 워치리스트 로드 후 정렬
watchlist = load_watchlist(watchlist_path)
ticker_info_cache = fetch_ticker_info(watchlist)  # marketCap 포함
sorted_watchlist = sort_by_market_and_cap(watchlist, ticker_info_cache)

# 기존 섹션 (정렬된 리스트 사용)
market_section = generate_market_overview()
analysis_section = generate_analysis(sorted_watchlist, ticker_info_cache)
ai_forecast_section = generate_ai_forecast_section(sorted_watchlist, ticker_info_cache)
extras_section = generate_extras(sorted_watchlist)
gem_section = generate_gem_scanner()

# 조합
html = market_section + analysis_section + ai_forecast_section + extras_section + gem_section
```

### 5. stock-analyzer 스킬 수정

**main.py:**

```python
import sys, os
sys.path.insert(0, os.path.expanduser("~/Developer/daily-stock-report/libs"))
from kronos_predictor import KronosPredictor

# --technical 플래그 시 기존 TA 실행 후 자동으로 Kronos도 실행
predictor = KronosPredictor()
forecast = predictor.predict(hist_df, horizons=[1, 5])
result["forecast"] = {
    "model": "Kronos-mini",
    **forecast,
    "disclaimer": "AI 예측은 참고용이며 투자 권유가 아닙니다."
}
```

**JSON 출력에 추가되는 forecast 키:**

```json
{
  "forecast": {
    "model": "Kronos-mini",
    "1d": {
      "median": 186.5,
      "p10": 183.2,
      "p90": 189.8,
      "direction_prob": 0.62,
      "volatility": 0.018
    },
    "5d": {
      "median": 190.1,
      "p10": 182.5,
      "p90": 197.3,
      "direction_prob": 0.58,
      "volatility": 0.032
    },
    "disclaimer": "AI 예측은 참고용이며 투자 권유가 아닙니다."
  }
}
```

**SKILL.md — Claude 해석 가이드 추가:**

Claude가 forecast JSON을 받으면 다음 형태로 사용자에게 제시:

```markdown
## 📈 AI 예측 (Kronos-mini)

| 기간 | 예상 범위 | 방향 | 상승 확률 | 변동성 |
|------|----------|------|---------|--------|
| 내일 | $183.2 ~ $189.8 | 🔼 | 62% | 보통 |
| 5일 | $182.5 ~ $197.3 | 🔼 | 58% | 높음 |

> ⚠ AI 예측은 참고용이며 투자 권유가 아닙니다.
```

## 타임아웃 전략

### GitHub Actions 전체 흐름

```
GitHub Actions 20분 타임아웃 (기존 15분 → 20분)
├── marketCap 조회 + 정렬: ~1분
├── market_overview: ~1분
├── analyzer (기존 TA): ~2분
├── ai_forecast (Kronos): 최대 10분 할당
│   ├── Phase 1: fallback 15종목 우선 처리 (~3분)
│   ├── Phase 2: 나머지 42종목 순차 처리
│   └── 7분 (70%) 도달 시 → 남은 종목 스킵
├── extras: ~1분
├── gem_scanner: ~1분
└── send_email: ~30초
```

### predict_batch 타임아웃 로직

1. `fallback_tickers` 15종목을 먼저 처리 (시총 상위 — 스킵되면 안 되는 핵심 종목)
2. 나머지 42종목 순차 처리
3. 매 종목 완료 시 경과 시간 체크
4. `경과 > timeout_seconds × 0.7` (기본 420초) 도달 → 중단
5. 미처리 종목은 `None` 반환 → HTML에 "⏱ 스킵" 표시

### fallback 종목

```python
FALLBACK_TICKERS = {
    # KOSPI 10
    "005930.KS",   # 삼성전자
    "000660.KS",   # SK하이닉스
    "005380.KS",   # 현대차
    "051910.KS",   # LG화학
    "373220.KS",   # LG에너지솔루션
    "006400.KS",   # 삼성SDI
    "035420.KS",   # 네이버
    "035720.KS",   # 카카오
    "017670.KS",   # SK텔레콤
    "259960.KS",   # 크래프톤
    # NASDAQ/NYSE 5
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
}
```

## 의존성 변경

### requirements.txt 추가

```
torch>=2.0.0
einops>=0.8.1
huggingface_hub>=0.33.1
safetensors>=0.6.2
```

Kronos 자체는 pip 패키지가 아니므로, 모델 가중치는 HuggingFace Hub에서 다운로드하고 추론 코드는 `kronos_predictor.py`에서 직접 구현한다. Kronos 레포의 `kronos/` 모듈 핵심 로직(토크나이저 + Transformer)을 래퍼에 포함.

### GitHub Actions workflow 수정

```yaml
timeout-minutes: 20  # 15 → 20

# pip install 단계에 torch 추가 (CPU only, 경량)
- run: pip install torch --index-url https://download.pytorch.org/whl/cpu
- run: pip install -r requirements.txt
```

CPU 전용 PyTorch로 설치하면 ~200MB (GPU 버전 ~2GB 대비 경량).

## 에러 처리

| 실패 지점 | 처리 |
|-----------|------|
| Kronos 모델 다운로드 실패 | ai_forecast 섹션 전체 스킵, 기존 리포트만 발송 |
| 개별 종목 predict 예외 | 해당 종목 None 처리, 다음 종목 계속 |
| 전체 타임아웃 도달 | 처리 완료된 종목만 표시, 스킵 카운트 표기 |
| yfinance 데이터 부족 (< 60일) | 해당 종목 Kronos 스킵, 기존 TA만 표시 |
| stock-analyzer에서 Kronos import 실패 | forecast 키 없이 기존 JSON만 반환 |

모든 에러에서 기존 리포트/스킬 기능은 영향받지 않는다 (graceful degradation).

## 범위 외 (향후 고려)

- Kronos fine-tuning (특정 시장 도메인 적응)
- GPU 환경에서 Kronos-base/large 사용
- 합성 데이터 생성을 활용한 전략 스트레스 테스트
- 기존 TA 시그널과 Kronos 확률을 결합한 앙상블 스코어링
- fundamental/ 모듈 구현 (현재 stub)
