# IntentMiner

🎯 **A production-grade pipeline to discover and propose missing intents in conversational AI systems.**

## Overview

IntentMiner automatically analyzes customer messages to discover hidden patterns within existing intent categories and proposes new, more granular intents to improve classification accuracy.

### Key Features

- **Intelligent Pattern Discovery** - LLM-powered semantic analysis + keyword clustering
- **Multi-Provider LLM Support** - OpenAI, Anthropic, and Google Gemini
- **Offline Mode** - Rule-based classification when API is unavailable
- **Robust Guardrails** - Min support thresholds, distinctiveness checks, fragmentation prevention
- **Comprehensive Reporting** - JSON proposals + Markdown analysis reports

## Quick Start

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Place input files in inputs/ directory

# 4. Run pipeline
python intent_expansion_pipeline.py --input inputs/ --output output/

# 5. Review results
cat output/ANALYSIS_REPORT.md
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for complete architecture and design documentation with Mermaid diagrams.

```mermaid
flowchart LR
    INPUT[📥 Input Files] --> S1[Stage 1: Ingestion]
    S1 --> S2[Stage 2: Classification]
    S2 --> S3[Stage 3: Grouping]
    S3 --> S4[Stage 4: Pattern Discovery]
    S4 --> S5[Stage 5: Proposals]
    S5 --> S6[Stage 6: Guardrails]
    S6 --> S7[Stage 7: Output]
    S7 --> OUTPUT[📤 Reports]
```

## Project Structure

```
IntentMiner/
├── intent_expansion_pipeline.py    # Main pipeline (1600+ lines)
├── config.yaml                     # Configuration
├── requirements.txt                # Dependencies
├── ARCHITECTURE.md                 # Full architecture docs
├── inputs/                         # Input data
│   ├── intent_mapper.json
│   ├── customer_messages.json
│   └── classification_prompt.txt
└── output/                         # Generated outputs
    ├── proposals.json
    ├── ANALYSIS_REPORT.md
    └── METRICS_REPORT.md
```

## License

MIT
