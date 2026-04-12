import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from sort_utils import sort_by_market_and_cap


class TestSortByMarketAndCap:

    def test_krx_before_us(self):
        tickers = [
            {"ticker": "AAPL", "name": "Apple", "market": "US"},
            {"ticker": "005930.KS", "name": "삼성전자", "market": "KRX"},
        ]
        cache = {
            "AAPL": {"marketCap": 3000000000000},
            "005930.KS": {"marketCap": 400000000000000},
        }
        result = sort_by_market_and_cap(tickers, cache)
        assert result[0]["ticker"] == "005930.KS"
        assert result[1]["ticker"] == "AAPL"

    def test_sort_by_cap_within_group(self):
        tickers = [
            {"ticker": "035720.KS", "name": "카카오", "market": "KRX"},
            {"ticker": "005930.KS", "name": "삼성전자", "market": "KRX"},
            {"ticker": "035420.KS", "name": "네이버", "market": "KRX"},
        ]
        cache = {
            "035720.KS": {"marketCap": 20000000000000},
            "005930.KS": {"marketCap": 400000000000000},
            "035420.KS": {"marketCap": 50000000000000},
        }
        result = sort_by_market_and_cap(tickers, cache)
        assert result[0]["ticker"] == "005930.KS"
        assert result[1]["ticker"] == "035420.KS"
        assert result[2]["ticker"] == "035720.KS"

    def test_missing_cap_goes_last(self):
        tickers = [
            {"ticker": "FIGM", "name": "Figma", "market": "US"},
            {"ticker": "AAPL", "name": "Apple", "market": "US"},
        ]
        cache = {
            "AAPL": {"marketCap": 3000000000000},
        }
        result = sort_by_market_and_cap(tickers, cache)
        assert result[0]["ticker"] == "AAPL"
        assert result[1]["ticker"] == "FIGM"

    def test_kosdaq_grouped_with_kospi(self):
        tickers = [
            {"ticker": "MSFT", "name": "Microsoft", "market": "US"},
            {"ticker": "035900.KQ", "name": "JYP", "market": "KRX"},
        ]
        cache = {
            "MSFT": {"marketCap": 3000000000000},
            "035900.KQ": {"marketCap": 5000000000000},
        }
        result = sort_by_market_and_cap(tickers, cache)
        assert result[0]["ticker"] == "035900.KQ"
