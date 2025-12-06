# IntentMiner - Architecture & Design Document

> **A production-grade pipeline to discover and propose missing intents in conversational AI systems**

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Pipeline Stages](#pipeline-stages)
5. [Core Components](#core-components)
6. [Data Flow](#data-flow)
7. [Configuration](#configuration)
8. [Input/Output Specifications](#inputoutput-specifications)
9. [Guardrails & Quality Control](#guardrails--quality-control)
10. [Deployment Guide](#deployment-guide)
11. [API Reference](#api-reference)

---

## Overview

**IntentMiner** is an intelligent pipeline that analyzes customer messages to discover hidden patterns within existing intent categories and proposes new, more granular intents to improve conversational AI classification accuracy.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Pattern Discovery** | LLM-powered semantic analysis combined with keyword clustering |
| **Scalable Processing** | Batch processing architecture supporting 200+ messages |
| **Multi-Provider LLM Support** | OpenAI, Anthropic, and Google Gemini integration |
| **Offline Mode** | Rule-based classification when API is unavailable |
| **Robust Guardrails** | Minimum support thresholds, distinctiveness checks, fragmentation prevention |
| **Comprehensive Reporting** | JSON proposals + Markdown analysis reports + Metrics dashboard |

---

## Problem Statement

### The Challenge

Conversational AI platforms classify customer intents to automate responses. However, current intent hierarchies are often **too broad**, leading to:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRENT STATE (PROBLEM)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Customer Message: "How to use this product?"                   │
│   Customer Message: "What's special about it?"                   │
│   Customer Message: "What are the ingredients?"                  │
│                                                                  │
│                           ↓                                      │
│                                                                  │
│   ALL classified as → "product_info" (too broad!)                │
│                                                                  │
│   Problems:                                                      │
│   ✗ Mixed customer needs under single intent                    │
│   ✗ Reduced personalization opportunities                       │
│   ✗ Lower automation accuracy                                   │
│   ✗ Missed analytics insights                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The Solution

```
┌─────────────────────────────────────────────────────────────────┐
│                    DESIRED STATE (SOLUTION)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Customer Message: "How to use this product?"                   │
│                    → "product_usage_instructions" ✓              │
│                                                                  │
│   Customer Message: "What's special about it?"                   │
│                    → "product_benefits_inquiry" ✓                │
│                                                                  │
│   Customer Message: "What are the ingredients?"                  │
│                    → "product_composition" ✓                     │
│                                                                  │
│   Benefits:                                                      │
│   ✓ Tailored responses for each specific need                   │
│   ✓ Better personalization and automation                       │
│   ✓ Improved classification accuracy                            │
│   ✓ Richer analytics and insights                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Solution Architecture

### High-Level Architecture Diagram

```mermaid
flowchart TB
    subgraph INPUT["📥 INPUT FILES"]
        IM[intent_mapper.json]
        CM[customer_messages.json]
        CP[classification_prompt.txt]
    end

    subgraph PIPELINE["⚙️ INTENTMINER PIPELINE"]
        direction TB
        S1["🔄 Stage 1<br/>Data Ingestion"]
        S2["🏷️ Stage 2<br/>Classification"]
        S3["📊 Stage 3<br/>Grouping"]
        S4["🔍 Stage 4<br/>Pattern Discovery"]
        S5["💡 Stage 5<br/>Proposal Generation"]
        S6["🛡️ Stage 6<br/>Guardrails"]
        S7["📝 Stage 7<br/>Output Generation"]
        
        S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    end

    subgraph LLM["🤖 LLM CLIENT"]
        direction LR
        OAI[OpenAI]
        ANT[Anthropic]
        GEM[Gemini]
        OFF[Offline Mode]
    end

    subgraph OUTPUT["📤 OUTPUT ARTIFACTS"]
        PJ[proposals.json]
        AR[ANALYSIS_REPORT.md]
        MR[METRICS_REPORT.md]
        MJ[metrics.json]
    end

    INPUT --> S1
    LLM -.-> S2
    LLM -.-> S4
    LLM -.-> S5
    S7 --> OUTPUT

    style INPUT fill:#e1f5fe
    style PIPELINE fill:#fff3e0
    style LLM fill:#f3e5f5
    style OUTPUT fill:#e8f5e9
```

### Pipeline Flow Detail

```mermaid
flowchart LR
    subgraph Stage1["Stage 1: Data Ingestion"]
        L1[Load JSON] --> N1[Normalize Text]
        N1 --> V1[Validate Data]
        V1 --> ST1[Generate Stats]
    end

    subgraph Stage2["Stage 2: Classification"]
        B2[Batch Messages] --> C2[Classify via LLM]
        C2 --> P2[Parse Response]
        P2 --> A2[Assign Intents]
    end

    subgraph Stage3["Stage 3: Grouping"]
        G3[Group by Intent] --> S3[Calculate Stats]
        S3 --> I3[Identify Patterns]
    end

    Stage1 --> Stage2 --> Stage3
```

```mermaid
flowchart LR
    subgraph Stage4["Stage 4: Pattern Discovery"]
        K4[Extract Keywords] --> CL4[Cluster Messages]
        CL4 --> LLM4[LLM Theme Analysis]
        LLM4 --> SC4[Score Patterns]
    end

    subgraph Stage5["Stage 5: Proposal Generation"]
        G5[Generate Intent] --> E5[Build Evidence]
        E5 --> Q5[Quality Score]
    end

    subgraph Stage6["Stage 6: Guardrails"]
        MS6[Min Support Check] --> TR6[Redundancy Check]
        TR6 --> MX6[Max Proposals Check]
        MX6 --> D6{Accept?}
    end

    Stage4 --> Stage5 --> Stage6
```

### Component Architecture

```mermaid
classDiagram
    class IntentExpansionPipeline {
        +PipelineConfig config
        +Logger logger
        +LLMClient llm_client
        +run(inputs_dir) Dict
        -_calculate_metrics() Dict
    }

    class PipelineConfig {
        +int min_support_absolute
        +float min_support_relative
        +str llm_provider
        +str llm_model
        +bool offline_mode
    }

    class LLMClient {
        +str provider
        +str model
        +query(prompt) str
        +classify_offline(message) Dict
    }

    class DataIngestionModule {
        +load_json(filepath) Dict
        +normalize_text(text) str
        +detect_language(text) str
        +ingest_and_prepare() Tuple
    }

    class ClassificationModule {
        +classify_batch(messages) List
        +classify_all(messages) List
    }

    class PatternDiscoveryModule {
        +extract_keywords(texts) List
        +discover_patterns_keyword_based() Dict
        +discover_patterns_llm_based() Dict
    }

    class ProposalGenerationModule {
        +generate_proposal(pattern) Dict
        +generate_all_proposals() List
    }

    class GuardrailsModule {
        +check_min_support(proposal) Tuple
        +check_terminology_redundancy() Tuple
        +apply_guardrails() Tuple
    }

    class OutputGenerationModule {
        +generate_proposals_json() str
        +generate_markdown_report() str
        +generate_metrics_report() str
    }

    IntentExpansionPipeline --> PipelineConfig
    IntentExpansionPipeline --> LLMClient
    IntentExpansionPipeline --> DataIngestionModule
    IntentExpansionPipeline --> ClassificationModule
    IntentExpansionPipeline --> PatternDiscoveryModule
    IntentExpansionPipeline --> ProposalGenerationModule
    IntentExpansionPipeline --> GuardrailsModule
    IntentExpansionPipeline --> OutputGenerationModule
    ClassificationModule --> LLMClient
    PatternDiscoveryModule --> LLMClient
    ProposalGenerationModule --> LLMClient
```

### Data Transformation Flow

```mermaid
flowchart TB
    subgraph Input["Raw Input"]
        RAW["200 Customer Messages<br/>+ Intent Mapper"]
    end

    subgraph Transform["Transformation Stages"]
        T1["Normalized Messages<br/>+ Metadata"]
        T2["Classified Messages<br/>+ Intent Labels"]
        T3["Grouped by<br/>(Primary, Secondary)"]
        T4["Discovered Patterns<br/>per Category"]
        T5["Generated Proposals<br/>(Unfiltered)"]
        T6["Validated Proposals<br/>(Accepted/Rejected)"]
    end

    subgraph Output["Final Output"]
        OUT["proposals.json<br/>ANALYSIS_REPORT.md<br/>METRICS_REPORT.md"]
    end

    RAW --> T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> OUT

    style Input fill:#ffecb3
    style Transform fill:#e3f2fd
    style Output fill:#c8e6c9
```

### Guardrails Decision Flow

```mermaid
flowchart TD
    START([New Proposal]) --> MS{Min Support<br/>≥ 10 messages?}
    MS -->|No| REJECT1[❌ Reject:<br/>Below threshold]
    MS -->|Yes| MR{Min Relative<br/>≥ 5%?}
    MR -->|No| REJECT2[❌ Reject:<br/>Low percentage]
    MR -->|Yes| TR{Terminology<br/>Unique?}
    TR -->|No| REJECT3[❌ Reject:<br/>Redundant name]
    TR -->|Yes| MP{Max Proposals<br/>< 15?}
    MP -->|No| REJECT4[❌ Reject:<br/>Limit reached]
    MP -->|Yes| ACCEPT[✅ Accept<br/>Proposal]

    style ACCEPT fill:#c8e6c9
    style REJECT1 fill:#ffcdd2
    style REJECT2 fill:#ffcdd2
    style REJECT3 fill:#ffcdd2
    style REJECT4 fill:#ffcdd2
```

### LLM Provider Strategy

```mermaid
flowchart TD
    REQ([LLM Request]) --> CHECK{API Key<br/>Available?}
    CHECK -->|No| OFFLINE[Use Offline Mode<br/>Rule-Based Classification]
    CHECK -->|Yes| PROVIDER{Select Provider}
    
    PROVIDER --> OAI[OpenAI<br/>gpt-4o-mini]
    PROVIDER --> ANT[Anthropic<br/>Claude]
    PROVIDER --> GEM[Google<br/>Gemini]
    
    OAI --> RETRY{Success?}
    ANT --> RETRY
    GEM --> RETRY
    
    RETRY -->|Yes| RESPONSE([Return Response])
    RETRY -->|No| BACKOFF[Exponential Backoff<br/>2^n seconds]
    BACKOFF --> ATTEMPT{Attempts<br/>< 3?}
    ATTEMPT -->|Yes| PROVIDER
    ATTEMPT -->|No| FALLBACK[Fallback to<br/>Offline Mode]
    
    OFFLINE --> RESPONSE
    FALLBACK --> RESPONSE

    style RESPONSE fill:#c8e6c9
    style OFFLINE fill:#fff9c4
    style FALLBACK fill:#fff9c4
```

### Technology Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.9+ |
| **LLM Providers** | OpenAI (gpt-4o-mini), Anthropic (Claude), Google Gemini |
| **Data Processing** | NumPy, Collections, Dataclasses |
| **Configuration** | YAML, Environment Variables |
| **Logging** | Python logging module |

---

## Pipeline Stages

### Stage 1: Data Ingestion & Preparation

**Purpose:** Load, validate, and normalize input data

```python
class DataIngestionModule:
    """
    Responsibilities:
    - Load JSON files (intent_mapper, customer_messages)
    - Normalize text (lowercase, strip whitespace)
    - Detect language (English, Hindi, etc.)
    - Generate dataset statistics
    """
```

**Process Flow:**

```mermaid
flowchart LR
    IM[intent_mapper.json] --> LOAD[Load & Parse]
    CM[customer_messages.json] --> LOAD
    CP[classification_prompt.txt] --> LOAD
    LOAD --> NORM[Normalize Text]
    NORM --> VAL[Validate Data]
    VAL --> STATS[Generate Stats]
    STATS --> OUT([Ready for<br/>Classification])

    style LOAD fill:#e3f2fd
    style NORM fill:#fff3e0
    style VAL fill:#f3e5f5
    style STATS fill:#e8f5e9
```

**Key Operations:**
- Text normalization: lowercase, strip, remove extra whitespace
- Language detection: English, Hindi (Devanagari), Chinese, Unknown
- Metadata enrichment: message ID, length, language

---

### Stage 2: Message Classification (Baseline)

**Purpose:** Classify all messages using the current intent taxonomy

```python
class ClassificationModule:
    """
    Responsibilities:
    - Use existing classification prompt + LLM
    - Assign (primary_intent, secondary_intent) to each message
    - Support batch processing for efficiency
    - Handle errors and uncertain classifications
    """
```

**Classification Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| **LLM-based** | Uses configured LLM provider | Production with API access |
| **Offline** | Rule-based keyword matching | Testing, no API available |

**Output per Message:**
```json
{
    "message_id": "msg_42",
    "primary": "specific_product",
    "secondary": "product_info",
    "confidence": 0.85
}
```

---

### Stage 3: Grouping & Organization

**Purpose:** Organize classified messages by intent pairs

```python
class GroupingModule:
    """
    Responsibilities:
    - Group messages by (primary, secondary) intent
    - Calculate group statistics
    - Identify under-represented groups
    """
```

**Group Statistics:**
- Message count per group
- Percentage distribution
- Average confidence scores
- Sample messages

---

### Stage 4: Pattern Discovery (Core Intelligence)

**Purpose:** Find hidden themes/patterns within each intent group

```python
class PatternDiscoveryModule:
    """
    Responsibilities:
    - Extract keyword frequencies & n-grams
    - Identify semantic clusters
    - Use LLM to discover sub-topics
    - Score pattern distinctiveness
    """
```

**Discovery Methods:**

#### Method 1: Keyword-Based (Lightweight, Deterministic)

```mermaid
flowchart LR
    subgraph Messages["Messages in 'product_info' group"]
        M1["'how to use?'"]
        M2["'daily routine?'"]
        M3["'how to apply?'"]
        M4["'what's special?'"]
        M5["'benefits?'"]
        M6["'is it effective?'"]
    end

    subgraph Patterns["Discovered Patterns"]
        P1["🔹 Pattern: 'usage'<br/>keywords: use, routine, apply"]
        P2["🔸 Pattern: 'benefits'<br/>keywords: special, benefit, effective"]
    end

    M1 --> P1
    M2 --> P1
    M3 --> P1
    M4 --> P2
    M5 --> P2
    M6 --> P2

    style P1 fill:#e3f2fd
    style P2 fill:#fff3e0
```

#### Method 2: LLM-Based (Intelligent, Semantic)

```mermaid
sequenceDiagram
    participant Pipeline
    participant LLM
    participant Parser

    Pipeline->>LLM: Analyze messages & identify themes
    Note over LLM: "What distinct themes<br/>do you see in these<br/>customer messages?"
    LLM-->>Parser: JSON Response
    Note over Parser: {themes: [<br/>  {name: "Usage", count: 12},<br/>  {name: "Benefits", count: 14}<br/>]}
    Parser-->>Pipeline: Structured Patterns
```

---

### Stage 5: Proposal Generation

**Purpose:** Create new intent proposals from discovered patterns

```python
class ProposalGenerationModule:
    """
    Responsibilities:
    - Generate intent name, ID, description
    - Align with existing taxonomy style
    - Calculate evidence metrics
    - Score proposal quality
    """
```

**Proposal Structure:**
```json
{
    "type": "new_secondary",
    "parent_primary_id": "specific_product",
    "parent_secondary_id": "product_info",
    "name": "Product Usage Instructions",
    "id": "product_usage",
    "description": "When customer asks how to use/apply a product",
    "evidence": {
        "message_count": 12,
        "percentage": "37.5%",
        "key_phrases": ["how", "use", "apply"],
        "sample_messages": ["How to use?", "Daily routine?"],
        "rationale": "Distinct from general info - requires instructions"
    },
    "quality_score": 0.89
}
```

---

### Stage 6: Guardrails & Validation

**Purpose:** Filter proposals to ensure quality and prevent issues

```python
class GuardrailsModule:
    """
    Responsibilities:
    - Minimum support threshold checks
    - Terminology redundancy detection
    - Over-fragmentation prevention
    - Semantic distinctiveness validation
    """
```

**Guardrail Checks:**

| Check | Threshold | Purpose |
|-------|-----------|---------|
| **Min Support (Absolute)** | ≥10 messages | Ensure statistical significance |
| **Min Support (Relative)** | ≥5% of parent | Ensure meaningful proportion |
| **Terminology Redundancy** | <50% token overlap | Prevent duplicate intents |
| **Max Total Proposals** | ≤15 | Prevent taxonomy explosion |
| **Max Per Category** | ≤3 | Prevent over-fragmentation |

---

### Stage 7: Output Generation

**Purpose:** Generate final deliverables

```python
class OutputGenerationModule:
    """
    Outputs:
    - proposals.json: Machine-readable proposals
    - ANALYSIS_REPORT.md: Human-readable findings
    - METRICS_REPORT.md: Pipeline assessment metrics
    - metrics.json: Raw metrics data
    - pipeline.log: Execution logs
    """
```

---

## Core Components

### LLMClient

**Multi-provider abstraction layer with built-in resilience:**

```python
class LLMClient:
    """
    Features:
    - Provider abstraction (OpenAI, Anthropic, Gemini)
    - Automatic retry with exponential backoff
    - Graceful fallback to offline mode
    - Configurable timeout and retries
    """
    
    # Supported Providers
    providers = ["openai", "anthropic", "gemini"]
    
    # Fallback Chain
    # 1. Try configured provider
    # 2. Retry with exponential backoff (2^n seconds)
    # 3. Fall back to offline mode (rule-based)
```

### RuleBasedClassifier

**Offline classification using keyword matching:**

```python
class RuleBasedClassifier:
    """
    Intent categories and keywords:
    
    basic_interactions:
      - greet: ["hi", "hello", "namaste"]
      - confirm: ["yes", "haan", "ji"]
      
    specific_product:
      - product_info: ["product", "what is", "details"]
      - how_to_use: ["how to use", "kaise use", "apply"]
      
    logistics:
      - order_status: ["track", "where is", "delivery"]
    """
```

### PipelineConfig

**Centralized configuration management:**

```python
@dataclass
class PipelineConfig:
    # Thresholds
    min_support_absolute: int = 10
    min_support_relative: float = 0.05
    max_patterns_per_category: int = 3
    max_total_proposals: int = 15
    
    # LLM Settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_batch_size: int = 50
    
    # Processing
    normalize_text: bool = True
    offline_mode: bool = False
```

---

## Data Flow

### Complete Data Transformation Pipeline

```mermaid
flowchart LR
    subgraph Stage1_2["Stages 1-2"]
        direction TB
        RAW["📥 Raw Messages<br/>(200 items)"]
        NORM["📝 Normalized<br/>+ Metadata"]
        CLASS["🏷️ Classified<br/>+ Intent Labels"]
        RAW --> NORM --> CLASS
    end

    subgraph Stage3_4["Stages 3-4"]
        direction TB
        GROUP["📊 Grouped<br/>by Intent"]
        PATTERN["🔍 Patterns<br/>Discovered"]
        GROUP --> PATTERN
    end

    subgraph Stage5_6["Stages 5-6"]
        direction TB
        PROP["💡 Raw<br/>Proposals"]
        FILTER["🛡️ Filtered<br/>Proposals"]
        PROP --> FILTER
    end

    subgraph Stage7["Stage 7"]
        OUT["📤 Final Outputs<br/>JSON + Markdown"]
    end

    Stage1_2 --> Stage3_4 --> Stage5_6 --> Stage7

    style Stage1_2 fill:#e3f2fd
    style Stage3_4 fill:#fff3e0
    style Stage5_6 fill:#f3e5f5
    style Stage7 fill:#e8f5e9
```

### Message Transformation Example

```mermaid
flowchart TB
    subgraph INPUT["📥 Input Message"]
        I1["current_human_message: 'How to use this product?'<br/>history: 'Looking at face serum'"]
    end

    subgraph S1["🔄 Stage 1: Normalization"]
        N1["id: 'msg_42'<br/>message: 'how to use this product?'<br/>length: 5<br/>language: 'english'"]
    end

    subgraph S2["🏷️ Stage 2: Classification"]
        C1["primary: 'specific_product'<br/>secondary: 'product_info'<br/>confidence: 0.85"]
    end

    subgraph S34["🔍 Stage 3-4: Pattern Found"]
        P1["name: 'Usage Instructions'<br/>count: 12<br/>keywords: ['how', 'use', 'apply']"]
    end

    subgraph S56["💡 Stage 5-6: Proposal"]
        PR1["name: 'Product Usage Instructions'<br/>id: 'product_usage'<br/>evidence: {count: 12, %: 37.5}"]
    end

    INPUT --> S1 --> S2 --> S34 --> S56

    style INPUT fill:#ffecb3
    style S1 fill:#e1f5fe
    style S2 fill:#e8f5e9
    style S34 fill:#fff3e0
    style S56 fill:#f3e5f5
```

---

## Configuration

### config.yaml Structure

```yaml
# Thresholds & Constraints
thresholds:
  min_support_absolute: 10      # Minimum messages for proposal
  min_support_relative: 0.05    # 5% of parent category
  semantic_distinctiveness_threshold: 0.6
  max_patterns_per_category: 3
  max_total_proposals: 15

# LLM Configuration
llm:
  provider: "openai"            # openai | anthropic | gemini
  model: "gpt-4o-mini"
  temperature: 0.3
  batch_size: 50
  timeout: 30
  max_retries: 3

# Text Processing
processing:
  normalize_text: true
  remove_duplicates: true
  detect_language: true
  handle_non_english: "skip"    # skip | translate | include
  min_message_length: 3

# Output Settings
output:
  directory: "./output/"
  log_level: "INFO"
```

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

---

## Input/Output Specifications

### Input Files

#### 1. intent_mapper.json
```json
{
    "intent_mapper": [
        {
            "primary_intent_name": "Specific Product",
            "primary_intent_id": "specific_product",
            "secondary_intents": [
                {
                    "name": "Product Info",
                    "id": "product_info",
                    "description": "When customer wants to know about a product"
                }
            ]
        }
    ]
}
```

#### 2. customer_messages.json
```json
{
    "customer_messages": [
        {
            "current_human_message": "How to use this product?",
            "history": "Looking at face serum"
        }
    ]
}
```

#### 3. classification_prompt.txt
```text
You are a customer service intent classifier.

History: {history}
Message: {message}

Available Intents:
{use_cases}

Classify the message and respond with JSON:
{"primary": "...", "secondary": "..."}
```

### Output Files

#### 1. proposals.json
```json
{
    "metadata": {
        "timestamp": "2025-12-06T10:00:00",
        "total_messages_analyzed": 200,
        "new_intents_proposed": 4
    },
    "proposed_intents": [
        {
            "name": "Product Usage Instructions",
            "id": "product_usage",
            "parent_primary_id": "specific_product",
            "parent_secondary_id": "product_info",
            "description": "...",
            "evidence": {...}
        }
    ]
}
```

#### 2. ANALYSIS_REPORT.md
Human-readable markdown report with findings, methodology, and recommendations.

#### 3. METRICS_REPORT.md
Pipeline assessment metrics including classification quality, coverage, and proposal scores.

---

## Guardrails & Quality Control

### Quality Scoring System

```mermaid
pie title Pipeline Quality Score (100 points)
    "Classification Quality" : 25
    "Taxonomy Coverage" : 20
    "Distribution Balance" : 15
    "Pattern Discovery" : 20
    "Proposal Generation" : 20
```

```mermaid
flowchart LR
    subgraph Scoring["🎯 Quality Score Components"]
        direction TB
        CQ["📊 Classification Quality<br/>25 points<br/><i>Certainty Rate</i>"]
        TC["📋 Taxonomy Coverage<br/>20 points<br/><i>% intents used</i>"]
        DB["⚖️ Distribution Balance<br/>15 points<br/><i>Gini coefficient</i>"]
        PD["🔍 Pattern Discovery<br/>20 points<br/><i>% groups with patterns</i>"]
        PG["💡 Proposal Generation<br/>20 points<br/><i>Quality proposals count</i>"]
    end

    subgraph Grade["📈 Final Grade"]
        A["A: 80-100 ✨"]
        B["B: 60-79 👍"]
        C["C: 40-59 ⚠️"]
        D["D: 0-39 ❌"]
    end

    CQ --> Grade
    TC --> Grade
    DB --> Grade
    PD --> Grade
    PG --> Grade

    style Scoring fill:#e3f2fd
    style Grade fill:#fff3e0
```

### Validation Rules

| Rule | Implementation | Failure Action |
|------|----------------|----------------|
| Min Message Count | `count >= 10` | Reject proposal |
| Min Percentage | `percentage >= 5%` | Reject proposal |
| Unique Name | `<50% overlap` | Reject proposal |
| Max Proposals | `total <= 15` | Stop accepting |
| Valid JSON | Schema validation | Log error, skip |

---

## Deployment Guide

### Quick Start

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Prepare input files in inputs/ directory

# 4. Run pipeline
python intent_expansion_pipeline.py --input inputs/ --output output/

# 5. Review results
cat output/ANALYSIS_REPORT.md
```

### Command Line Options

```bash
python intent_expansion_pipeline.py [OPTIONS]

Options:
  --config FILE    Config file path (default: config.yaml)
  --input DIR      Input directory (default: inputs/)
  --output DIR     Output directory (default: output/)
  --offline        Run in offline mode (no API required)
```

### Running in Offline Mode

```bash
# No API key required - uses rule-based classification
python intent_expansion_pipeline.py --offline
```

---

## API Reference

### Main Classes

| Class | Purpose |
|-------|---------|
| `IntentExpansionPipeline` | Main orchestrator |
| `PipelineConfig` | Configuration dataclass |
| `LLMClient` | LLM provider abstraction |
| `DataIngestionModule` | Stage 1: Data loading |
| `ClassificationModule` | Stage 2: Classification |
| `GroupingModule` | Stage 3: Grouping |
| `PatternDiscoveryModule` | Stage 4: Pattern detection |
| `ProposalGenerationModule` | Stage 5: Proposal creation |
| `GuardrailsModule` | Stage 6: Validation |
| `OutputGenerationModule` | Stage 7: Report generation |

### Usage Example

```python
from intent_expansion_pipeline import IntentExpansionPipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    min_support_absolute=10,
    output_dir="./output/"
)

# Run pipeline
pipeline = IntentExpansionPipeline(config)
result = pipeline.run("./inputs/")

print(f"Status: {result['status']}")
print(f"Proposals: {result['proposals']}")
```

---

## Project Structure

```
exported-assets/
├── intent_expansion_pipeline.py    # Main pipeline implementation (1600+ lines)
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── ARCHITECTURE.md                 # This document
│
├── inputs/                         # Input data directory
│   ├── intent_mapper.json          # Current intent hierarchy
│   ├── customer_messages.json      # Customer messages to analyze
│   └── classification_prompt.txt   # Classification prompt template
│
└── output/                         # Generated outputs
    ├── proposals.json              # Machine-readable proposals
    ├── ANALYSIS_REPORT.md          # Human-readable analysis
    ├── METRICS_REPORT.md           # Pipeline metrics
    ├── metrics.json                # Raw metrics data
    └── pipeline.log                # Execution logs
```

---


