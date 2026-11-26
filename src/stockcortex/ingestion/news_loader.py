from datetime import datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import feedparser
import httpx
import pandas as pd

from stockcortex.config import STARTER_UNIVERSE, settings
from stockcortex.storage.duck_store import DuckStore
from stockcortex.storage.sqlite_ledger import SQLiteLedger

IST = ZoneInfo("Asia/Kolkata")
NSE_CLOSE_TIME = time(15, 30)


def map_published_to_market_date(published_utc: datetime) -> str:
    """
    Maps a UTC publication timestamp to the earliest NSE trading date whose close
    has NOT yet occurred at publication time (prevents post-15:30 IST look-ahead leakage).
    """
    if published_utc.tzinfo is None:
        published_utc = published_utc.replace(tzinfo=timezone.utc)
    dt_ist = published_utc.astimezone(IST)
    market_day = dt_ist.date()

    # If published at or after 15:30 IST, it can only inform the next trading day
    if dt_ist.time() >= NSE_CLOSE_TIME:
        market_day += timedelta(days=1)

    # Roll forward weekend dates (Saturday=5 -> Monday, Sunday=6 -> Monday)
    while market_day.weekday() >= 5:
        market_day += timedelta(days=1)

    return market_day.isoformat()


class EntityNewsLoader:
    """Fetches Indian financial news via NewsData.io with Google News India RSS fallback."""

    def __init__(
        self,
        duck_store: DuckStore | None = None,
        ledger: SQLiteLedger | None = None,
        bronze_dir: Path | None = None,
        silver_dir: Path | None = None,
    ) -> None:
        self.duck = duck_store or DuckStore()
        self.ledger = ledger or SQLiteLedger()
        self.bronze_dir = bronze_dir or settings.bronze_dir
        self.silver_dir = silver_dir or settings.silver_dir

    @staticmethod
    def build_entity_keywords(symbol: str, company_name: str | None = None) -> list[str]:
        clean_sym = symbol.upper().strip()
        full_name = company_name or STARTER_UNIVERSE.get(clean_sym, clean_sym)
        short_name = full_name.replace(" Ltd", "").replace(" Limited", "").strip()
        keywords = {clean_sym.lower(), full_name.lower(), short_name.lower()}
        first_word = short_name.split()[0].lower()
        if len(first_word) >= 4 and first_word not in {"state", "bank", "india"}:
            keywords.add(first_word)
        return sorted(keywords)

    def _fetch_newsdata_io(self, query: str) -> list[dict[str, Any]]:
        api_key = settings.newsdata_api_key
        if not api_key:
            return []
        url = "https://newsdata.io/api/1/latest"
        params = {
            "apikey": api_key,
            "q": query,
            "country": "in",
            "language": "en",
        }
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()

        articles = []
        for item in payload.get("results", []) or []:
            pub_str = item.get("pubDate")
            if not pub_str:
                continue
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            articles.append(
                {
                    "title": item.get("title", ""),
                    "description": item.get("description") or "",
                    "source_name": item.get("source_id") or "newsdata.io",
                    "url": item.get("link") or "",
                    "published_utc": pub_dt.astimezone(timezone.utc).isoformat(),
                    "provider": "newsdata_io",
                }
            )
        return articles

    def _fetch_google_news_rss(self, query: str) -> list[dict[str, Any]]:
        encoded_q = quote_plus(f"{query} NSE stock")
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        articles = []
        for entry in getattr(feed, "entries", []):
            pub_raw = getattr(entry, "published", "")
            if not pub_raw:
                continue
            try:
                pub_dt = parsedate_to_datetime(pub_raw)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
            source_obj = getattr(entry, "source", {})
            source_title = source_obj.get("title", "google_news_rss") if isinstance(source_obj, dict) else "google_news_rss"
            articles.append(
                {
                    "title": getattr(entry, "title", ""),
                    "description": getattr(entry, "summary", ""),
                    "source_name": source_title,
                    "url": getattr(entry, "link", ""),
                    "published_utc": pub_dt.astimezone(timezone.utc).isoformat(),
                    "provider": "google_news_rss",
                }
            )
        return articles

    def filter_and_align_articles(self, raw_articles: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
        keywords = self.build_entity_keywords(symbol)
        rows = []
        for art in raw_articles:
            haystack = f"{art.get('title', '')} {art.get('description', '')}".lower()
            if not any(kw in haystack for kw in keywords):
                continue
            pub_dt = datetime.fromisoformat(art["published_utc"])
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "market_date": map_published_to_market_date(pub_dt),
                    "published_utc": pub_dt.astimezone(timezone.utc).isoformat(),
                    "title": art["title"].strip(),
                    "description": art.get("description", "").strip(),
                    "source_name": art.get("source_name", ""),
                    "url": art.get("url", ""),
                    "provider": art.get("provider", ""),
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=["symbol", "market_date", "published_utc", "title", "description", "source_name", "url", "provider"]
            )

        df = pd.DataFrame(rows).drop_duplicates(subset=["symbol", "title"]).sort_values("published_utc").reset_index(drop=True)
        return df

    def ingest_symbol_news(self, symbol: str) -> Path:
        clean_sym = symbol.upper().strip()
        company_name = STARTER_UNIVERSE.get(clean_sym, clean_sym)
        short_query = company_name.replace(" Ltd", "").strip()

        raw_articles: list[dict[str, Any]] = []
        provider_used = "newsdata_io"
        try:
            raw_articles = self._fetch_newsdata_io(short_query)
        except Exception as exc:
            self.ledger.log_ingestion("BRONZE_NEWS", clean_sym, "newsdata_io", "FALLBACK_TRIGGERED", 0, str(exc))

        if not raw_articles:
            provider_used = "google_news_rss"
            raw_articles = self._fetch_google_news_rss(short_query)

        # Save immutable Bronze JSON snapshot
        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        bronze_json_path = self.bronze_dir / f"{clean_sym}_news_{ts_tag}.json"
        bronze_json_path.write_text(json.dumps(raw_articles, indent=2, ensure_ascii=False), encoding="utf-8")

        silver_df = self.filter_and_align_articles(raw_articles, symbol=clean_sym)
        silver_path = self.silver_dir / f"{clean_sym}_news.parquet"

        # Merge with any existing Silver news cache to accumulate history cleanly
        if silver_path.exists():
            existing_df = self.duck.read_parquet(silver_path)
            silver_df = (
                pd.concat([existing_df, silver_df], ignore_index=True)
                .drop_duplicates(subset=["symbol", "title"])
                .sort_values("published_utc")
                .reset_index(drop=True)
            )

        self.duck.write_parquet(silver_df, silver_path)
        self.ledger.log_ingestion(
            layer="SILVER_NEWS",
            symbol=clean_sym,
            source_provider=provider_used,
            status="SUCCESS",
            rows_processed=len(silver_df),
            anomaly_details=f"raw_fetched={len(raw_articles)}; relevant_total={len(silver_df)}",
        )
        return silver_path
