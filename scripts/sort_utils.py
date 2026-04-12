"""시장/시총 기반 워치리스트 정렬 유틸리티."""

import yfinance as yf
import logging

logger = logging.getLogger(__name__)

_KRX_SUFFIXES = ('.KS', '.KQ')


def fetch_market_caps(tickers):
    """워치리스트의 시가총액을 일괄 조회.

    Args:
        tickers: [{"ticker": "005930.KS", ...}, ...]

    Returns:
        {ticker_str: {"marketCap": int}}. 조회 실패 시 해당 키 없음.
    """
    cache = {}
    for stock in tickers:
        ticker = stock["ticker"]
        try:
            info = yf.Ticker(ticker).info
            cap = info.get("marketCap")
            if cap and cap > 0:
                cache[ticker] = {"marketCap": cap}
        except Exception as e:
            logger.warning("시총 조회 실패 %s: %s", ticker, e)
    return cache


def sort_by_market_and_cap(tickers, ticker_info_cache=None):
    """시장 그룹 (KRX → US) + 시총 내림차순 정렬.

    Args:
        tickers: [{"ticker": "005930.KS", "name": "삼성전자", "market": "KRX"}, ...]
        ticker_info_cache: {ticker: {"marketCap": int}}. None이면 yfinance 조회.

    Returns:
        정렬된 새 리스트 (원본 불변).
    """
    if ticker_info_cache is None:
        ticker_info_cache = fetch_market_caps(tickers)

    def sort_key(stock):
        ticker = stock["ticker"]
        is_krx = 0 if ticker.endswith(_KRX_SUFFIXES) else 1
        cap = ticker_info_cache.get(ticker, {}).get("marketCap", 0)
        return (is_krx, -cap)

    return sorted(tickers, key=sort_key)
