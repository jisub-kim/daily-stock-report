"""
ai_forecast.py 단위 테스트.
모든 테스트는 mock을 사용하여 실제 Kronos 모델 로딩 없이 실행.
"""

import sys
import os

import pytest

# scripts/ 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from ai_forecast import (
    _format_price,
    _format_range,
    _direction_icon,
    _volatility_label,
    build_forecast_html,
)


# ---------------------------------------------------------------------------
# Test 1: _format_price
# ---------------------------------------------------------------------------

class TestFormatPrice:
    def test_krw_no_decimals(self):
        """KRW는 소수점 없이 천 단위 구분자 사용."""
        result = _format_price(58500, "KRW")
        assert result == "₩58,500", f"KRW 포맷 오류: {result}"

    def test_krw_large_number(self):
        """KRW 큰 수 포맷."""
        result = _format_price(1234567, "KRW")
        assert result == "₩1,234,567"

    def test_usd_two_decimals(self):
        """USD는 소수점 2자리와 $ 기호 사용."""
        result = _format_price(138.25, "USD")
        assert result == "$138.25", f"USD 포맷 오류: {result}"

    def test_usd_integer_price(self):
        """USD 정수 가격은 .00으로 표시."""
        result = _format_price(100.0, "USD")
        assert result == "$100.00"

    def test_non_krw_defaults_to_dollar(self):
        """KRW 외 통화는 $ 기호를 사용."""
        result = _format_price(50.5, "EUR")
        assert result.startswith("$")


# ---------------------------------------------------------------------------
# Test 1.5: _format_range
# ---------------------------------------------------------------------------

class TestFormatRange:
    def test_krw_range(self):
        result = _format_range(58000.0, 59000.0, "KRW")
        assert "58,000" in result
        assert "59,000" in result

    def test_usd_range(self):
        result = _format_range(170.0, 180.0, "USD")
        assert "$" in result
        assert "170" in result
        assert "180" in result


# ---------------------------------------------------------------------------
# Test 2: _direction_icon
# ---------------------------------------------------------------------------

class TestDirectionIcon:
    def test_up_threshold(self):
        """확률 0.55 이상이면 상승 아이콘."""
        assert _direction_icon(0.55) == "🔼"
        assert _direction_icon(0.8) == "🔼"
        assert _direction_icon(1.0) == "🔼"

    def test_down_threshold(self):
        """확률 0.45 이하이면 하락 아이콘."""
        assert _direction_icon(0.45) == "🔽"
        assert _direction_icon(0.2) == "🔽"
        assert _direction_icon(0.0) == "🔽"

    def test_sideways(self):
        """확률 0.46~0.54이면 횡보 아이콘."""
        assert _direction_icon(0.5) == "➡️"
        assert _direction_icon(0.46) == "➡️"
        assert _direction_icon(0.54) == "➡️"


# ---------------------------------------------------------------------------
# Test 3: _volatility_label
# ---------------------------------------------------------------------------

class TestVolatilityLabel:
    def test_high_volatility(self):
        """변동성 0.03 초과이면 '높음'."""
        assert _volatility_label(0.031) == "높음"
        assert _volatility_label(0.1) == "높음"

    def test_medium_volatility(self):
        """변동성 0.015~0.03이면 '보통'."""
        assert _volatility_label(0.015) == "보통"
        assert _volatility_label(0.02) == "보통"
        assert _volatility_label(0.03) == "보통"

    def test_low_volatility(self):
        """변동성 0.015 미만이면 '낮음'."""
        assert _volatility_label(0.014) == "낮음"
        assert _volatility_label(0.0) == "낮음"


# ---------------------------------------------------------------------------
# Test 4: build_forecast_html — 정상 케이스
# ---------------------------------------------------------------------------

_MOCK_FORECASTS = {
    "AAPL": {
        1: {"median": 175.0, "p10": 170.0, "p90": 180.0, "direction_prob": 0.65, "volatility": 0.02},
        5: {"median": 178.0, "p10": 168.0, "p90": 188.0, "direction_prob": 0.6, "volatility": 0.025},
    },
    "MSFT": {
        1: {"median": 420.0, "p10": 415.0, "p90": 425.0, "direction_prob": 0.4, "volatility": 0.015},
        5: {"median": 422.0, "p10": 410.0, "p90": 435.0, "direction_prob": 0.45, "volatility": 0.018},
    },
}

_MOCK_STOCKS = [
    {"ticker": "AAPL", "name": "Apple", "market": "US", "price": 172.0, "currency": "USD"},
    {"ticker": "MSFT", "name": "Microsoft", "market": "US", "price": 418.0, "currency": "USD"},
]


class TestBuildForecastHtml:
    def test_returns_html_string(self):
        """정상 forecasts가 있으면 HTML 문자열 반환."""
        html = build_forecast_html(_MOCK_FORECASTS, _MOCK_STOCKS)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_contains_stock_names(self):
        """결과 HTML에 종목명이 포함되어야 함."""
        html = build_forecast_html(_MOCK_FORECASTS, _MOCK_STOCKS)
        assert "Apple" in html
        assert "Microsoft" in html

    def test_contains_tickers(self):
        """결과 HTML에 티커가 포함되어야 함."""
        html = build_forecast_html(_MOCK_FORECASTS, _MOCK_STOCKS)
        assert "AAPL" in html
        assert "MSFT" in html

    def test_contains_1d_section(self):
        """1일 예측 섹션 라벨이 있어야 함."""
        html = build_forecast_html(_MOCK_FORECASTS, _MOCK_STOCKS)
        assert "1일 예측" in html

    def test_contains_5d_section(self):
        """5일(주간) 예측 섹션 라벨이 있어야 함."""
        html = build_forecast_html(_MOCK_FORECASTS, _MOCK_STOCKS)
        assert "5일(주간) 예측" in html

    def test_contains_disclaimer(self):
        """AI 예측 면책 문구가 있어야 함."""
        html = build_forecast_html(_MOCK_FORECASTS, _MOCK_STOCKS)
        assert "AI 예측은 참고용" in html

    def test_contains_direction_icons(self):
        """상승/하락 아이콘이 HTML에 포함되어야 함."""
        html = build_forecast_html(_MOCK_FORECASTS, _MOCK_STOCKS)
        # AAPL direction_prob=0.65 → 🔼
        assert "🔼" in html
        # MSFT 1d direction_prob=0.4 → 🔽
        assert "🔽" in html

    def test_contains_current_price(self):
        """현재가가 HTML에 표시되어야 함."""
        html = build_forecast_html(_MOCK_FORECASTS, _MOCK_STOCKS)
        assert "$172.00" in html
        assert "$418.00" in html

    def test_contains_price_range(self):
        """예상 범위가 HTML에 표시되어야 함 (USD: $p10~$p90)."""
        html = build_forecast_html(_MOCK_FORECASTS, _MOCK_STOCKS)
        assert "$170.0~$180.0" in html

    def test_contains_probability(self):
        """상승확률이 % 형식으로 HTML에 표시되어야 함."""
        html = build_forecast_html(_MOCK_FORECASTS, _MOCK_STOCKS)
        # AAPL direction_prob=0.65 → "65%"
        assert "65%" in html

    def test_contains_volatility_label(self):
        """변동성 라벨이 HTML에 표시되어야 함."""
        html = build_forecast_html(_MOCK_FORECASTS, _MOCK_STOCKS)
        # AAPL 5d volatility=0.025 → "보통"
        assert "보통" in html


# ---------------------------------------------------------------------------
# Test 5: 스킵된 종목 처리
# ---------------------------------------------------------------------------

class TestSkippedTicker:
    def test_skipped_ticker_shows_skip_text(self):
        """None 예측 결과 종목은 '스킵' 텍스트를 표시해야 함."""
        forecasts = {
            "AAPL": _MOCK_FORECASTS["AAPL"],
            "MSFT": None,  # 스킵된 종목
        }
        stocks = _MOCK_STOCKS
        html = build_forecast_html(forecasts, stocks)
        assert "스킵" in html, f"스킵 텍스트가 HTML에 없음: {html[:300]}"

    def test_skipped_ticker_shows_timer_icon(self):
        """스킵 종목에는 ⏱ 아이콘이 있어야 함."""
        forecasts = {"AAPL": None}
        stocks = [{"ticker": "AAPL", "name": "Apple", "market": "US", "price": 172.0, "currency": "USD"}]
        html = build_forecast_html(forecasts, stocks)
        assert "⏱" in html

    def test_skip_count_in_footer(self):
        """스킵 종목이 있으면 푸터에 스킵 개수가 표시되어야 함."""
        forecasts = {
            "AAPL": _MOCK_FORECASTS["AAPL"],
            "MSFT": None,
        }
        html = build_forecast_html(forecasts, _MOCK_STOCKS)
        assert "스킵됨" in html, "스킵 개수 안내가 없음"

    def test_no_skip_count_when_all_processed(self):
        """모든 종목이 처리되었으면 스킵 개수 안내가 없어야 함."""
        html = build_forecast_html(_MOCK_FORECASTS, _MOCK_STOCKS)
        assert "스킵됨" not in html


# ---------------------------------------------------------------------------
# Test 6: 빈 forecasts는 "" 반환
# ---------------------------------------------------------------------------

class TestEmptyForecasts:
    def test_empty_forecasts_returns_empty_string(self):
        """forecasts가 빈 dict이면 빈 문자열 반환."""
        html = build_forecast_html({}, _MOCK_STOCKS)
        assert html == "", f"빈 forecasts에서 '' 기대, 실제: {repr(html)}"

    def test_no_matching_stocks_returns_empty_string(self):
        """forecasts에 stocks 티커가 없으면 빈 문자열 반환."""
        forecasts = {"NVDA": _MOCK_FORECASTS["AAPL"]}  # stocks에 없는 티커
        html = build_forecast_html(forecasts, _MOCK_STOCKS)
        assert html == "", f"매칭 없는 forecasts에서 '' 기대, 실제: {repr(html)}"


# ---------------------------------------------------------------------------
# Test 7: KRW 종목 포맷 검증
# ---------------------------------------------------------------------------

class TestKrwFormatting:
    def test_krw_price_range_format(self):
        """KRW 종목의 예상 범위에 ₩ 없이 숫자만 표시되어야 함."""
        forecasts = {
            "005930.KS": {
                1: {"median": 58500.0, "p10": 58000.0, "p90": 59000.0, "direction_prob": 0.6, "volatility": 0.02},
                5: {"median": 59000.0, "p10": 57000.0, "p90": 61000.0, "direction_prob": 0.55, "volatility": 0.025},
            }
        }
        stocks = [{"ticker": "005930.KS", "name": "삼성전자", "market": "KRX", "price": 58200.0, "currency": "KRW"}]
        html = build_forecast_html(forecasts, stocks)
        assert "58,000~59,000" in html, f"KRW 범위 포맷 오류. HTML: {html[:500]}"

    def test_krw_current_price_format(self):
        """KRW 현재가에 ₩ 기호가 포함되어야 함."""
        forecasts = {
            "005930.KS": {
                1: {"median": 58500.0, "p10": 58000.0, "p90": 59000.0, "direction_prob": 0.6, "volatility": 0.02},
                5: {"median": 59000.0, "p10": 57000.0, "p90": 61000.0, "direction_prob": 0.55, "volatility": 0.025},
            }
        }
        stocks = [{"ticker": "005930.KS", "name": "삼성전자", "market": "KRX", "price": 58200.0, "currency": "KRW"}]
        html = build_forecast_html(forecasts, stocks)
        assert "₩58,200" in html
