"""Find public forecasts (prediction markets) related to a question.

FutureEval rules allow bots to use any forecast that is publicly available to
human forecasters, so matching markets are handed to the forecaster as evidence.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import requests
from forecasting_tools import GeneralLlm, MetaculusQuestion, clean_indents

logger = logging.getLogger(__name__)

POLYMARKET_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
MANIFOLD_MARKET_URL = "https://api.manifold.markets/v0/market/{id}"
HTTP_TIMEOUT = 15
MIN_MANIFOLD_TRADERS = 10  # thinner play-money markets are mostly noise
MAX_OUTCOMES_SHOWN = 8


async def find_public_forecasts(question: MetaculusQuestion, query_llm: GeneralLlm) -> str:
    """Return one line per related market, or "" when nothing was found."""
    queries = await _search_queries(question, query_llm)
    lines: list[str] = []
    seen_urls: set[str] = set()
    for query in queries:
        results = await asyncio.gather(
            asyncio.to_thread(_search_polymarket, query),
            asyncio.to_thread(_search_manifold, query),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                logger.warning(f"Market search failed for {query!r}: {result}")
                continue
            for url, line in result:
                if url not in seen_urls:
                    seen_urls.add(url)
                    lines.append(line)
    return "\n".join(lines)


async def _search_queries(question: MetaculusQuestion, llm: GeneralLlm) -> list[str]:
    prompt = clean_indents(
        f"""
        Write up to 3 short search queries (2-6 words each) that would find prediction
        markets about the same event as this forecasting question. Use names and key
        terms a market title would contain. Reply with only a JSON list of strings.

        Question: {question.question_text}
        """
    )
    try:
        reply = await llm.invoke(prompt)
        queries = json.loads(reply[reply.index("[") : reply.rindex("]") + 1])
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if queries:
            return queries[:3]
    except Exception as e:  # the fallback below is good enough to keep going
        logger.warning(f"Query generation failed, using the question title: {e}")
    return [question.question_text[:120]]


def _search_polymarket(query: str) -> list[tuple[str, str]]:
    response = requests.get(
        POLYMARKET_SEARCH_URL,
        params={"q": query, "limit_per_type": 5, "events_status": "active"},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    found = []
    for event in response.json().get("events") or []:
        prices = []
        for market in event.get("markets") or []:
            if market.get("closed"):
                continue
            label = market.get("groupItemTitle") or market.get("question") or "?"
            price = _polymarket_price_text(market)
            if price:
                prices.append(f"{label}: {price}")
        if not prices:
            continue
        url = f"https://polymarket.com/event/{event.get('slug')}"
        volume = float(event.get("volume") or 0)
        ends = (event.get("endDate") or "?")[:10]
        shown = "; ".join(prices[:MAX_OUTCOMES_SHOWN])
        more = f" (+{len(prices) - MAX_OUTCOMES_SHOWN} more)" if len(prices) > MAX_OUTCOMES_SHOWN else ""
        found.append(
            (url, f"- Polymarket (real money, volume ${volume:,.0f}, ends {ends}): "
                  f"{event.get('title')} | {shown}{more} | {url}")
        )
    return found


def _polymarket_price_text(market: dict) -> str | None:
    try:
        outcomes = json.loads(market.get("outcomes") or "[]")
        prices = [float(p) for p in json.loads(market.get("outcomePrices") or "[]")]
    except (TypeError, ValueError):
        return None
    if not outcomes or len(outcomes) != len(prices):
        return None
    if "Yes" in outcomes:
        return f"{prices[outcomes.index('Yes')]:.0%} Yes"
    return ", ".join(f"{o} {p:.0%}" for o, p in zip(outcomes, prices))


def _search_manifold(query: str) -> list[tuple[str, str]]:
    response = requests.get(
        MANIFOLD_SEARCH_URL,
        params={"term": query, "limit": 5, "filter": "open"},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    found = []
    for market in response.json():
        traders = market.get("uniqueBettorCount") or 0
        if traders < MIN_MANIFOLD_TRADERS:
            continue
        if market.get("outcomeType") == "BINARY":
            odds = f"{market['probability']:.0%} Yes"
        elif market.get("outcomeType") == "MULTIPLE_CHOICE":
            odds = _manifold_answers_text(market["id"])
        else:
            continue
        if not odds:
            continue
        found.append(
            (market["url"], f"- Manifold (play money, {traders} traders, closes "
                            f"{_date_from_ms(market.get('closeTime'))}): {market['question']} | "
                            f"{odds} | {market['url']}")
        )
    return found


def _manifold_answers_text(market_id: str) -> str | None:
    response = requests.get(MANIFOLD_MARKET_URL.format(id=market_id), timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    answers = sorted(
        (a for a in response.json().get("answers") or [] if a.get("probability") is not None),
        key=lambda a: a["probability"],
        reverse=True,
    )
    if not answers:
        return None
    return "; ".join(f"{a['text']}: {a['probability']:.0%}" for a in answers[:MAX_OUTCOMES_SHOWN])


def _date_from_ms(timestamp_ms: int | None) -> str:
    if not timestamp_ms:
        return "?"
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return "?"
