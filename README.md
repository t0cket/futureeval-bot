# FutureEval forecasting bot

A bot for the [Metaculus FutureEval](https://www.metaculus.com/futureeval/) bot tournaments. It builds on Metaculus's official Fall 2026 template, from the `forecasting-tools` package.

## How it works

For each new question:

1. **Research.** Three sources run in parallel:
   - **News:** AskNews latest and historical news summaries
   - **Web search:** Claude Opus 5.5 with live web search, reporting the latest developments, scheduled events before the resolution date, base rates, and whether the question is already nearly settled
   - **Related public forecasts:** matching Polymarket and Manifold markets, with current prices, volume and close dates. Search queries are written by Claude Haiku 4.5. Manifold markets with fewer than 10 traders are dropped.
2. **Forecast.** Claude Opus 5.5 at high effort writes 3 independent forecasts from the same research. The binary and multiple-choice prompts ask for the time left, the status-quo outcome, a base rate with its reference class, and what related markets say. A market is used only if it asks about the same event with compatible criteria and timing.
3. **Aggregate.** The 3 forecasts are combined with the template's standard aggregation (median for binary questions), then posted with the reasoning as a private comment.

Numeric, date and conditional questions use the template's prompts with the same research.

## Rules this bot follows

- **No human in the loop.** Forecasts are posted only by the scheduled workflow and are never edited by hand.
- **Testing only on safe questions.** Changes are tested on the `bot-testing-area` tournament or closed questions, never on open tournament questions.
- **One forecast per question.** Questions it has already forecast are skipped.
- **Public sources only.** The only forecasts used as evidence are ones any human could see.

## Setup

1. **Metaculus**
   - Log in with your human account and open Settings → My forecasting bots.
   - Create a bot and copy its token.
   - Fill in the [participation form](https://forms.gle/aQdYMq9Pisrf1v7d8). Its second section requests LLM credits.
2. **OpenRouter**
   - Either use the key Metaculus sends after the form, or create your own key at openrouter.ai.
   - If you use your own key, set a credit limit on it. The bot skips runs once less than $1 is left.
3. **AskNews (optional, free)**
   - Sign up at my.asknews.app with your Metaculus email.
   - Then email contact@asknews.app asking for the FutureEval bot allowance, and create a client ID and secret.
4. **GitHub**
   - Push this folder to a new **public** repository. Public repos get unlimited Actions minutes; the 20-minute schedule would exceed a private repo's free quota.
   - Under Settings → Secrets and variables → Actions, add `METACULUS_TOKEN`, `OPENROUTER_API_KEY` and, if you have them, `ASKNEWS_CLIENT_ID` and `ASKNEWS_SECRET`.
5. **Test**
   - Run Actions → Test bot → Run workflow.
   - On Metaculus, use "Switch to bot account" to check that forecasts landed on the bot-testing-area questions.
   - After that, the "Forecast on tournament questions" workflow runs every 20 minutes.

To skip MiniBench, set the repository variable `INCLUDE_MINIBENCH` to `0`. To change the number of samples per question, set `PREDICTIONS_PER_QUESTION`.

## Local runs

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows; use .venv/bin/pip elsewhere
cp .env.template .env                           # then fill in your keys
.venv/Scripts/python bot.py --mode test --dry-run
```
