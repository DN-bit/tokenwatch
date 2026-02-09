# ⛽ TokenWatch

**Self-contained LLM router for AI agents.**

Single-file Python router that distributes requests across multiple LLM providers based on rate limits, cost, and task complexity.

## Features

- 🔀 **Smart Routing**: Automatically fails over when providers hit rate limits
- 💰 **Cost Optimization**: Routes simple tasks to cheaper models
- 📊 **Live Dashboard**: Real-time usage tracking and cost monitoring
- 💾 **Persistent Logging**: SQLite database for usage analytics
- 🔌 **OpenAI-Compatible API**: Drop-in replacement for standard endpoints
- 📦 **Single File**: No dependencies beyond Python standard library

## Quick Start

```bash
# 1. Set API keys
export ANTHROPIC_API_KEY="sk-..."
export KIMI_API_KEY="..."
export OPENAI_API_KEY="sk-..."

# 2. Run TokenWatch
python tokenwatch.py

# 3. Open dashboard
open http://localhost:8080
```

## API Usage

```bash
# Route automatically based on complexity
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4-5",
    "messages": [{"role": "user", "content": "Hello"}],
    "complexity": "simple"
  }'
```

**Complexity levels:**
- `simple` → Prefer cheaper models (Kimi, GPT-4o-mini)
- `complex` → Prefer capable models (Claude, GPT-4o)

## Dashboard

Visit `http://localhost:8080` for real-time monitoring:
- Rate limit status across providers
- Token usage and costs
- Response times
- Error tracking

## Configuration

Set environment variables for API keys:
- `ANTHROPIC_API_KEY`
- `KIMI_API_KEY`
- `OPENAI_API_KEY`

Or edit the `PROVIDERS` dict in `tokenwatch.py`.

## How It Works

1. **Request comes in** with optional `complexity` hint
2. **Router selects** best provider based on:
   - Current rate limit usage
   - Task complexity
   - Cost optimization
3. **Request routed** to selected provider
4. **Usage logged** to database with cost calculation
5. **Response returned** with routing metadata

## Rate Limit Logic

- Provider used only if under 80% of rate limit
- Falls back to next provider automatically
- All providers saturated → waits or errors gracefully

## Supported Models

| Provider | Models | Rate Limit |
|----------|--------|------------|
| Anthropic | claude-opus-4-5, sonnet-4-5, haiku-4-5 | 30K-50K tokens/min |
| Kimi | kimi-k2.5, kimi-k1.5 | 60K tokens/min |
| OpenAI | gpt-4o, gpt-4o-mini | 10K-20K tokens/min |

## Open Source

MIT License - Fork and customize for your needs.

Built for agents, by agents. 🤖
