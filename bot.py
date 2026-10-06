"""Forecasting bot for the Metaculus FutureEval bot tournaments.

Builds on Metaculus's official Fall 2026 template bot and changes three things:
1. Research combines AskNews, live web search and related prediction markets.
2. Forecasts come from Claude Opus 5.5 at high effort, several samples per question.
3. Binary and multiple-choice prompts ask for a base rate and say how to weigh
   market prices.

Usage:
    python bot.py                      # seasonal tournament (+ MiniBench unless INCLUDE_MINIBENCH=0)
    python bot.py --mode test          # bot-testing-area questions, safe for smoke tests
    python bot.py --mode test --dry-run
"""
import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

import dotenv
import requests

dotenv.load_dotenv()

from forecasting_tools import (  # noqa: E402  (env must be loaded first)
    AskNewsSearcher,
    BinaryQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
)
from forecasting_tools.forecast_bots.official_bots.template_bot_2026_fall import (  # noqa: E402
    FallTemplateBot2026,
)

from public_forecasts import find_public_forecasts  # noqa: E402

logger = logging.getLogger(__name__)

FORECAST_MODEL = os.getenv("FORECAST_MODEL", "openrouter/anthropic/claude-opus-5.5")
FORECAST_EFFORT = os.getenv("FORECAST_EFFORT", "high")
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "openrouter/anthropic/claude-opus-5.5:online")
PARSER_MODEL = os.getenv("PARSER_MODEL", "openrouter/anthropic/claude-haiku-4.5")
PREDICTIONS_PER_QUESTION = int(os.getenv("PREDICTIONS_PER_QUESTION", "3"))
INCLUDE_MINIBENCH = os.getenv("INCLUDE_MINIBENCH", "1") == "1"
MIN_CREDITS_USD = float(os.getenv("MIN_CREDITS_USD", "1.0"))

MARKET_GUIDANCE = (
    "Use a listed prediction market only if it asks about essentially the same event with "
    "compatible resolution criteria and timing. A high-volume real-money market is strong "
    "evidence: move far from its price only for a specific reason. Play-money markets with "
    "few traders are weak evidence."
)


class CalibratedForecaster(FallTemplateBot2026):
    """Official Fall 2026 template with richer research and market-aware prompts."""

    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            news, web, markets = await asyncio.gather(
                self._news(question),
                self._web_research(question),
                find_public_forecasts(question, self.get_llm("parser", "llm")),
                return_exceptions=True,
            )
        research = "\n\n".join(
            f"## {title}\n{_section_text(body)}"
            for title, body in [
                ("News (AskNews)", news),
                ("Web research", web),
                ("Related public forecasts", markets),
            ]
        )
        logger.info(f"Research for {question.page_url}:\n{research}")
        return research

    async def _news(self, question: MetaculusQuestion) -> str:
        if not (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")):
            return ""
        return await AskNewsSearcher().get_formatted_news_async(question.question_text)

    async def _web_research(self, question: MetaculusQuestion) -> str:
        prompt = clean_indents(
            f"""
            You are a research assistant to a superforecaster. Search the web and report the
            most relevant, current facts for the question below. You do not forecast.

            Cover:
            - The latest developments, with dates and sources
            - Scheduled events before the resolution date that could decide the outcome
            - Base rates or historical frequencies for comparable events
            - Whether the question may already be resolved, or nearly determined, under its criteria

            Question:
            {question.question_text}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print}

            Today is {datetime.now().strftime("%Y-%m-%d")}.
            """
        )
        return await self.get_llm("researcher", "llm").invoke(prompt)

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Research (news, web search and related prediction markets):
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering, write:
            (a) The time left until the outcome is known.
            (b) The status quo outcome if nothing changed.
            (c) A base rate: how often comparable situations resolved Yes, and the reference class you used.
            (d) What related prediction markets say, if any are listed. {MARKET_GUIDANCE}
            (e) The strongest argument for Yes and the strongest argument for No.

            Then combine these into a probability. Good forecasters put extra weight on the status quo because the world changes slowly most of the time. When the evidence is strong, commit to it: being confidently right scores far better than hedging toward 50%.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Research (news, web search and related prediction markets):
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering, write:
            (a) The time left until the outcome is known.
            (b) The status quo outcome if nothing changed.
            (c) Base rates for the options, and the reference class you used.
            (d) What related prediction markets say, if any are listed. {MARKET_GUIDANCE}
            (e) A scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            Good forecasters put extra weight on the status quo, and leave some probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)


def _section_text(body: str | BaseException) -> str:
    if isinstance(body, BaseException):
        return f"Unavailable ({type(body).__name__}: {body})"
    return body.strip() or "None found."


def openrouter_credits_remaining() -> float | None:
    """Credits left on the OpenRouter key, or None when the key has no limit."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    response = requests.get(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()["data"]
    logger.info(f"OpenRouter key usage so far: ${data.get('usage', 0):.2f}")
    remaining = data.get("limit_remaining")
    return None if remaining is None else float(remaining)


def build_bot(publish: bool, skip_previously_forecasted: bool) -> CalibratedForecaster:
    return CalibratedForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=PREDICTIONS_PER_QUESTION,
        use_research_summary_to_forecast=False,
        enable_summarize_research=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to="reports",
        skip_previously_forecasted_questions=skip_previously_forecasted,
        extra_metadata_in_explanation=True,
        llms={
            "default": GeneralLlm(
                model=FORECAST_MODEL,
                timeout=300,
                allowed_tries=2,
                max_tokens=16000,
                extra_body={"reasoning": {"effort": FORECAST_EFFORT}},
            ),
            "researcher": GeneralLlm(
                model=RESEARCH_MODEL, timeout=300, allowed_tries=2, max_tokens=8000
            ),
            "parser": GeneralLlm(model=PARSER_MODEL, timeout=60, allowed_tries=3),
            "summarizer": GeneralLlm(model=PARSER_MODEL, timeout=60, allowed_tries=3),
        },
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(description="Run the FutureEval forecasting bot")
    parser.add_argument("--mode", choices=["tournament", "test"], default="tournament")
    parser.add_argument("--dry-run", action="store_true", help="forecast without publishing")
    args = parser.parse_args()

    if not os.getenv("METACULUS_TOKEN"):
        sys.exit("METACULUS_TOKEN is not set (see README.md)")
    remaining = openrouter_credits_remaining()
    if remaining is not None and remaining < MIN_CREDITS_USD:
        logger.warning(f"Only ${remaining:.2f} of LLM credits left; skipping this run.")
        return

    client = MetaculusClient()
    if args.mode == "test":
        bot = build_bot(publish=not args.dry_run, skip_previously_forecasted=False)
        tournaments = ["bot-testing-area"]
    else:
        bot = build_bot(publish=not args.dry_run, skip_previously_forecasted=True)
        tournaments = [client.CURRENT_AI_COMPETITION_ID]
        if INCLUDE_MINIBENCH:
            tournaments.append(client.CURRENT_MINIBENCH_ID)

    reports = asyncio.run(forecast_all(bot, tournaments))
    bot.log_report_summary(reports)


async def forecast_all(bot: CalibratedForecaster, tournaments: list) -> list:
    # One event loop for all tournaments: the class-level semaphore binds to the
    # first loop that waits on it.
    reports = []
    for tournament in tournaments:
        reports += await bot.forecast_on_tournament(tournament, return_exceptions=True)
    return reports


if __name__ == "__main__":
    main()
