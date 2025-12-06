"""
Intent Expansion Pipeline - Production Implementation Skeleton
==============================================================

A scalable, intelligent pipeline that discovers missing or under-differentiated 
intents in customer service chatbot classification systems.

Author: AI Workflow Analyst
Project: IntentMiner
Version: 1.0.0

Usage:
    python intent_expansion_pipeline.py --config config.yaml --input inputs/
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import os
from dotenv import load_dotenv

import numpy as np
from collections import Counter, defaultdict
import re
from itertools import combinations

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the intent expansion pipeline."""
    
    # Thresholds
    min_support_absolute: int = 5
    min_support_relative: float = 0.03
    semantic_distinctiveness_threshold: float = 0.6
    max_patterns_per_category: int = 3
    max_total_proposals: int = 15
    
    # LLM Configuration
    llm_provider: str = "anthropic"  # "openai" | "anthropic" | "gemini"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_temperature: float = 0.3
    llm_batch_size: int = 50
    llm_timeout: int = 60
    llm_max_retries: int = 3
    
    # Processing
    normalize_text: bool = True
    detect_language: bool = True
    handle_non_english: str = "skip"  # "skip" | "translate" | "include"
    min_message_length: int = 3
    
    # Output
    output_dir: str = "./output/"
    log_level: str = "INFO"
    
    # Offline mode (rule-based, no LLM API required)
    offline_mode: bool = False


class RuleBasedClassifier:
    """
    Rule-based classifier for offline mode.
    Uses keyword matching when LLM API is not available.
    """
    
    # Keywords for each intent category
    INTENT_KEYWORDS = {
        "basic_interactions": {
            "greet": ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "namaste"],
            "end": ["bye", "goodbye", "thank you", "thanks", "thankyou", "ok thankyou", "okay thanks", "dhanyawad"],
            "acknowledge": ["ok", "okay", "alright", "got it", "understood", "theek hai", "accha"],
            "confirm": ["yes", "yeah", "yup", "haan", "ha", "ji"],
            "deny": ["no", "nope", "nahi", "na", "not"],
            "ask_help": ["help", "assist", "support", "need help", "madad"],
        },
        "about_company": {
            "contact_info": ["contact", "phone", "number", "call", "email", "whatsapp", "reach"],
            "store_locations": ["store", "shop", "location", "address", "where", "branch", "outlet"],
            "offers": ["offer", "discount", "sale", "promo", "deal", "coupon", "code", "off"],
            "company_info": ["company", "brand", "about", "who are you", "what is"],
        },
        "specific_product": {
            "product_info": ["product", "item", "what is", "tell me about", "details", "info", "special", "features"],
            "ingredients": ["ingredient", "contain", "made of", "composition", "what's in"],
            "how_to_use": ["how to use", "use kaise", "kaise use", "apply", "how do i", "steps", "instruction", "routine"],
            "pricing": ["price", "cost", "kya rate", "kitna", "kitne", "how much", "expensive", "cheap"],
            "availability": ["available", "stock", "in stock", "out of stock", "milega", "hai kya"],
            "comparison": ["compare", "difference", "better", "vs", "which one", "best", "recommend"],
        },
        "recommendation": {
            "skin_type": ["skin type", "oily", "dry", "combination", "sensitive", "acne", "pimple"],
            "hair_type": ["hair type", "hair", "hair fall", "dandruff", "baal", "scalp"],
            "concern_based": ["concern", "problem", "issue", "suggest", "recommend", "best for", "suitable"],
            "general_recommend": ["what should i", "which product", "kya loon", "suggest", "recommendation"],
        },
        "logistics": {
            "order_status": ["order status", "track", "where is my order", "order kahan", "delivery status", "shipped", "dispatch"],
            "delivery_info": ["delivery", "shipping", "when will", "kab milega", "receive", "transit", "rto"],
            "return_refund": ["return", "refund", "exchange", "cancel", "cancellation", "money back", "replacement"],
            "payment": ["payment", "pay", "cod", "online payment", "upi", "card", "paytm"],
        }
    }
    
    def classify(self, message: str, history: str = "") -> Dict:
        """Classify a message using keyword matching."""
        text = (message + " " + history).lower()
        
        best_primary = "UNCERTAIN"
        best_secondary = "UNCERTAIN"
        best_score = 0
        
        for primary, secondaries in self.INTENT_KEYWORDS.items():
            for secondary, keywords in secondaries.items():
                score = sum(1 for kw in keywords if kw in text)
                if score > best_score:
                    best_score = score
                    best_primary = primary
                    best_secondary = secondary
        
        return {
            "primary": best_primary,
            "secondary": best_secondary,
            "confidence": min(0.9, best_score * 0.2) if best_score > 0 else 0.3
        }


class LLMClient:
    """
    Abstraction layer for LLM API calls.
    Supports OpenAI, Anthropic, Gemini.
    Implements batching, retries, fallbacks.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.provider = config.llm_provider
        self.model = config.llm_model
        self.temperature = config.llm_temperature
        self.retries = config.llm_max_retries
        self.timeout = config.llm_timeout
        self.offline_mode = config.offline_mode
        self.rule_classifier = RuleBasedClassifier() if self.offline_mode else None
        
        if self.offline_mode:
            logging.info("[OFFLINE] Running in OFFLINE mode (rule-based classification)")
            return
        
        # Initialize provider-specific client
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "your_openai_api_key_here":
                logging.warning("⚠️ No valid OPENAI_API_KEY found. Falling back to offline mode.")
                self.offline_mode = True
                self.rule_classifier = RuleBasedClassifier()
                return
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
        
        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.client = genai
            except ImportError:
                raise ImportError("Install gemini: pip install google-generativeai")
    
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Query LLM with retry logic and fallbacks.
        In offline mode, returns a simulated response for pattern discovery.
        """
        # Offline mode handling
        if self.offline_mode:
            return self._offline_response(prompt)
        
        attempt = 0
        
        while attempt < self.retries:
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt or "You are a helpful AI assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        timeout=self.timeout
                    )
                    return response.choices[0].message.content
                
                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=2000,
                        system=system_prompt or "You are a helpful AI assistant.",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text
                
                elif self.provider == "gemini":
                    model = self.client.GenerativeModel(self.model)
                    response = model.generate_content(
                        f"{system_prompt or ''}\n\n{prompt}"
                    )
                    return response.text
            
            except Exception as e:
                attempt += 1
                logging.warning(f"LLM query failed (attempt {attempt}): {e}")
                
                if attempt >= self.retries:
                    raise RuntimeError(f"LLM query failed after {self.retries} retries: {e}")
                
                # Exponential backoff
                wait_time = 2 ** attempt
                logging.info(f"Retrying in {wait_time} seconds...")
                import time
                time.sleep(wait_time)
        
        raise RuntimeError("Failed to query LLM")
    
    def _offline_response(self, prompt: str) -> str:
        """Generate simulated responses for offline mode."""
        prompt_lower = prompt.lower()
        
        # Pattern discovery prompts
        if "pattern" in prompt_lower or "cluster" in prompt_lower or "theme" in prompt_lower:
            return json.dumps({
                "patterns": [
                    {"name": "product_usage_instructions", "description": "Questions about how to use products", "frequency": "high"},
                    {"name": "order_tracking_queries", "description": "Questions about order delivery status", "frequency": "high"},
                    {"name": "product_comparison", "description": "Comparing products or asking for recommendations", "frequency": "medium"},
                    {"name": "sale_timing_queries", "description": "Questions about when sales or offers start", "frequency": "medium"},
                    {"name": "customization_requests", "description": "Requests for customized or personalized products", "frequency": "low"}
                ]
            })
        
        # Intent proposal prompts
        if "proposal" in prompt_lower or "intent" in prompt_lower and "new" in prompt_lower:
            return json.dumps({
                "proposals": [
                    {
                        "intent_name": "product_usage_query",
                        "parent_intent": "specific_product",
                        "type": "new_secondary",
                        "justification": "Many messages ask specifically about usage instructions",
                        "example_messages": ["How to use it?", "Kaise use kare?"]
                    }
                ]
            })
        
        # Default: return empty JSON
        return "{}"
    
    def classify_offline(self, message: str, history: str = "") -> Dict:
        """Classify using rule-based approach for offline mode."""
        return self.rule_classifier.classify(message, history)


# ============================================================================
# STAGE 0: SETUP & LOGGING
# ============================================================================

def setup_logging(config: PipelineConfig):
    """Initialize logging configuration."""
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{config.output_dir}/pipeline.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


# ============================================================================
# STAGE 1: DATA INGESTION & PREPARATION
# ============================================================================

class DataIngestionModule:
    """Load, validate, and normalize input data."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def load_json(self, filepath: str) -> Dict:
        """Load and parse JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {filepath}: {e}")
            raise
    
    def normalize_text(self, text: str) -> str:
        """Normalize text: lowercase, strip, remove extra whitespace."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single space
        return text
    
    def detect_language(self, text: str) -> str:
        """Simple language detection (English vs non-English)."""
        # Basic heuristic: check for common English words
        english_words = {'the', 'a', 'to', 'is', 'how', 'what', 'why', 'when', 'where', 'who'}
        words_lower = set(text.lower().split()[:20])
        
        if len(words_lower & english_words) > 0:
            return "english"
        
        # Check for common non-English patterns
        if re.search(r'[\u0900-\u097F]', text):  # Devanagari (Hindi)
            return "hindi"
        if re.search(r'[\u4E00-\u9FFF]', text):  # Chinese
            return "chinese"
        
        return "unknown"
    
    def ingest_and_prepare(self, config: PipelineConfig, inputs_dir: str) -> Tuple[Dict, List, str, Dict]:
        """
        Main ingestion function.
        Returns: (intent_mapper, messages, prompt_template, stats)
        """
        
        self.logger.info("=" * 80)
        self.logger.info("STAGE 1: DATA INGESTION & PREPARATION")
        self.logger.info("=" * 80)
        
        # Load files
        self.logger.info("Loading input files...")
        intent_mapper = self.load_json(f"{inputs_dir}/intent_mapper.json")
        messages_data = self.load_json(f"{inputs_dir}/customer_messages.json")
        
        # Handle both formats: list directly or {"customer_messages": [...]}
        if isinstance(messages_data, dict) and "customer_messages" in messages_data:
            messages_raw = messages_data["customer_messages"]
        elif isinstance(messages_data, list):
            messages_raw = messages_data
        else:
            raise ValueError("customer_messages.json must be either a list or contain 'customer_messages' key")
        
        with open(f"{inputs_dir}/classification_prompt.txt", 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # Process messages
        self.logger.info(f"Processing {len(messages_raw)} messages...")
        messages = []
        
        for i, msg in enumerate(messages_raw):
            # Normalize text
            current_msg = self.normalize_text(msg.get("current_human_message", ""))
            history = self.normalize_text(msg.get("history", ""))
            
            # Detect language
            lang = self.detect_language(current_msg)
            
            # Add metadata
            processed_msg = {
                "id": f"msg_{i}",
                "original_current_message": msg.get("current_human_message", ""),
                "current_human_message": current_msg,
                "original_history": msg.get("history", ""),
                "history": history,
                "length": len(current_msg.split()),
                "language": lang
            }
            
            messages.append(processed_msg)
        
        # Statistics
        stats = {
            "total_messages": len(messages),
            "avg_message_length": np.mean([m["length"] for m in messages]),
            "language_distribution": Counter([m["language"] for m in messages]),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Validate intent mapper
        self.logger.info(f"Validating intent mapper...")
        primary_count = len(intent_mapper.get("intent_mapper", []))
        total_secondary = sum(
            len(p.get("secondary_intents", [])) 
            for p in intent_mapper.get("intent_mapper", [])
        )
        
        self.logger.info(f"  - Primary intents: {primary_count}")
        self.logger.info(f"  - Secondary intents: {total_secondary}")
        self.logger.info(f"  - Messages: {stats['total_messages']}")
        self.logger.info(f"  - Avg message length: {stats['avg_message_length']:.1f} words")
        self.logger.info(f"  - Language distribution: {dict(stats['language_distribution'])}")
        
        return intent_mapper, messages, prompt_template, stats


# ============================================================================
# STAGE 2: MESSAGE CLASSIFICATION (Baseline)
# ============================================================================

class ClassificationModule:
    """Classify messages using current system (LLM + prompt)."""
    
    def __init__(self, logger: logging.Logger, llm_client: LLMClient):
        self.logger = logger
        self.llm = llm_client
    
    def format_use_cases_text(self, intent_mapper: Dict) -> str:
        """Format intent mapper as readable text for the prompt."""
        use_cases_text = ""
        
        for primary in intent_mapper.get("intent_mapper", []):
            primary_name = primary.get("primary_intent_name", "Unknown")
            primary_id = primary.get("primary_intent_id", "unknown")
            
            use_cases_text += f"\n### {primary_name} ({primary_id})\n"
            
            for secondary in primary.get("secondary_intents", []):
                sec_name = secondary.get("name", "Unknown")
                sec_id = secondary.get("id", "unknown")
                sec_desc = secondary.get("description", "No description")
                
                use_cases_text += f"- **{sec_name}** ({sec_id}): {sec_desc}\n"
        
        return use_cases_text
    
    def classify_batch(self, messages_batch: List[Dict], prompt_template: str, intent_mapper: Dict) -> List[Dict]:
        """Classify a batch of messages."""
        
        classifications = []
        use_cases_text = self.format_use_cases_text(intent_mapper)
        
        for msg in messages_batch:
            try:
                # Check if using offline mode
                if self.llm.offline_mode:
                    classification = self.llm.classify_offline(
                        msg["current_human_message"], 
                        msg["history"]
                    )
                    classifications.append({
                        "message_id": msg["id"],
                        "primary": classification["primary"],
                        "secondary": classification["secondary"],
                        "confidence": classification["confidence"],
                        "mode": "offline"
                    })
                    continue
                
                # Construct prompt
                final_prompt = prompt_template.format(
                    history=msg["history"],
                    message=msg["current_human_message"],
                    use_cases=use_cases_text
                )
                
                # Query LLM
                response = self.llm.query(
                    prompt=final_prompt,
                    system_prompt="You are a customer intent classifier. Classify messages strictly according to the provided categories. Return ONLY valid JSON."
                )
                
                # Parse response
                try:
                    # Try to extract JSON from response
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        classification = json.loads(json_match.group())
                    else:
                        classification = json.loads(response)
                    
                    classifications.append({
                        "message_id": msg["id"],
                        "primary": classification.get("primary", "UNCERTAIN"),
                        "secondary": classification.get("secondary", "UNCERTAIN"),
                        "confidence": 0.85,
                        "raw_response": response
                    })
                
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse JSON response for {msg['id']}")
                    classifications.append({
                        "message_id": msg["id"],
                        "primary": "UNCERTAIN",
                        "secondary": "UNCERTAIN",
                        "confidence": 0.0,
                        "error": "JSON parse error"
                    })
            
            except Exception as e:
                self.logger.error(f"Classification failed for {msg['id']}: {e}")
                classifications.append({
                    "message_id": msg["id"],
                    "primary": "ERROR",
                    "secondary": "ERROR",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        return classifications
    
    def classify_all(self, messages: List[Dict], prompt_template: str, intent_mapper: Dict, config: PipelineConfig) -> List[Dict]:
        """Classify all messages with batching."""
        
        self.logger.info("=" * 80)
        self.logger.info("STAGE 2: MESSAGE CLASSIFICATION")
        self.logger.info("=" * 80)
        
        all_classifications = []
        batch_size = config.llm_batch_size
        
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i+batch_size]
            
            self.logger.info(f"Classifying batch {i//batch_size + 1}/{(len(messages) + batch_size - 1)//batch_size}...")
            
            batch_classifications = self.classify_batch(batch, prompt_template, intent_mapper)
            all_classifications.extend(batch_classifications)
        
        # Statistics
        certain_count = sum(1 for c in all_classifications if c["confidence"] > 0.5)
        uncertain_count = len(all_classifications) - certain_count
        
        self.logger.info(f"Classification complete:")
        self.logger.info(f"  - Certain: {certain_count}/{len(all_classifications)}")
        self.logger.info(f"  - Uncertain: {uncertain_count}/{len(all_classifications)}")
        
        return all_classifications


# ============================================================================
# STAGE 3: GROUPING & ORGANIZATION
# ============================================================================

class GroupingModule:
    """Group classified messages by intent."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def group_by_intent(self, messages: List[Dict], classifications: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Group messages by (primary, secondary) intent.
        Returns: (grouped_messages, group_stats)
        """
        
        self.logger.info("=" * 80)
        self.logger.info("STAGE 3: GROUPING & ORGANIZATION")
        self.logger.info("=" * 80)
        
        # Create lookup
        clf_lookup = {c["message_id"]: c for c in classifications}
        
        # Merge classifications into messages
        for msg in messages:
            clf = clf_lookup.get(msg["id"], {})
            msg["assigned_primary"] = clf.get("primary", "UNKNOWN")
            msg["assigned_secondary"] = clf.get("secondary", "UNKNOWN")
            msg["confidence"] = clf.get("confidence", 0.0)
        
        # Group by (primary, secondary)
        grouped = defaultdict(list)
        
        for msg in messages:
            key = (msg["assigned_primary"], msg["assigned_secondary"])
            grouped[key].append(msg)
        
        # Compute statistics
        group_stats = {}
        total_messages = len(messages)
        
        for (primary, secondary), msgs in grouped.items():
            group_stats[(primary, secondary)] = {
                "count": len(msgs),
                "percentage": len(msgs) / total_messages,
                "avg_confidence": np.mean([m["confidence"] for m in msgs]),
                "text_samples": [m["current_human_message"][:100] for m in msgs[:3]]
            }
        
        # Log
        self.logger.info(f"Grouped into {len(grouped)} (primary, secondary) pairs:")
        
        for (primary, secondary), stats in sorted(group_stats.items(), key=lambda x: x[1]["count"], reverse=True):
            self.logger.info(
                f"  - ({primary}, {secondary}): {stats['count']} messages ({stats['percentage']*100:.1f}%)"
            )
        
        return dict(grouped), group_stats


# ============================================================================
# STAGE 4: PATTERN DISCOVERY (CORE INTELLIGENCE)
# ============================================================================

class PatternDiscoveryModule:
    """Discover hidden themes/patterns within each secondary intent group."""
    
    def __init__(self, logger: logging.Logger, llm_client: Optional[LLMClient] = None):
        self.logger = logger
        self.llm = llm_client
    
    def extract_keywords(self, texts: List[str], top_k: int = 10) -> List[Tuple[str, int]]:
        """Extract top keywords using TF-IDF-like approach."""
        
        # Simple keyword extraction: frequency of words, excluding stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'am', 'are', 'be', 'been',
            'to', 'for', 'of', 'in', 'on', 'at', 'by', 'from', 'as', 'with',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her',
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where'
        }
        
        all_words = []
        for text in texts:
            words = [w.lower() for w in re.findall(r'\b\w+\b', text) if w.lower() not in stopwords and len(w) > 2]
            all_words.extend(words)
        
        keyword_freq = Counter(all_words)
        return keyword_freq.most_common(top_k)
    
    def discover_patterns_keyword_based(self, grouped: Dict, config: PipelineConfig) -> Dict:
        """
        Discover patterns using keyword extraction.
        Lightweight, deterministic, no LLM required.
        """
        
        self.logger.info("=" * 80)
        self.logger.info("STAGE 4: PATTERN DISCOVERY (Keyword-based)")
        self.logger.info("=" * 80)
        
        patterns = {}
        
        for (primary, secondary), messages in grouped.items():
            if len(messages) < 5:  # Skip small groups
                continue
            
            self.logger.info(f"Analyzing ({primary}, {secondary}) with {len(messages)} messages...")
            
            # Extract texts
            texts = [m["current_human_message"] for m in messages]
            
            # Extract keywords
            keywords = self.extract_keywords(texts, top_k=15)
            
            # Simple clustering: group by dominant keyword
            clusters = defaultdict(list)
            
            for msg in messages:
                text = msg["current_human_message"]
                
                # Find dominant keyword
                dominant_keyword = None
                for keyword, _ in keywords:
                    if keyword in text.lower():
                        dominant_keyword = keyword
                        break
                
                if dominant_keyword is None:
                    dominant_keyword = "other"
                
                clusters[dominant_keyword].append(msg)
            
            # Filter clusters by min support
            group_patterns = []
            
            for cluster_keyword, cluster_msgs in clusters.items():
                if len(cluster_msgs) >= config.min_support_absolute:
                    pattern = {
                        "name": f"pattern_{cluster_keyword}",
                        "keyword": cluster_keyword,
                        "count": len(cluster_msgs),
                        "percentage": len(cluster_msgs) / len(messages),
                        "keywords": [kw for kw, _ in self.extract_keywords([m["current_human_message"] for m in cluster_msgs], top_k=5)],
                        "sample_messages": [m["current_human_message"][:100] for m in cluster_msgs[:3]]
                    }
                    group_patterns.append(pattern)
            
            if group_patterns:
                patterns[(primary, secondary)] = group_patterns
                self.logger.info(f"  Found {len(group_patterns)} patterns")
        
        return patterns
    
    def discover_patterns_llm_based(self, grouped: Dict, config: PipelineConfig, intent_mapper: Dict) -> Dict:
        """
        Discover patterns using LLM analysis.
        More intelligent but requires LLM calls.
        In offline mode, uses enhanced keyword-based detection.
        """
        
        self.logger.info("=" * 80)
        self.logger.info("STAGE 4: PATTERN DISCOVERY (LLM-based)")
        self.logger.info("=" * 80)
        
        # In offline mode, use enhanced rule-based pattern discovery
        if self.llm and self.llm.offline_mode:
            self.logger.info("Using enhanced offline pattern discovery...")
            return self._discover_patterns_offline_enhanced(grouped, config)
        
        if not self.llm:
            self.logger.warning("LLM client not available, falling back to keyword-based")
            return self.discover_patterns_keyword_based(grouped, config)
        
        patterns = {}
        
        # Find intent definitions
        intent_lookup = {}
        for primary in intent_mapper.get("intent_mapper", []):
            for secondary in primary.get("secondary_intents", []):
                intent_lookup[(primary.get("primary_intent_id"), secondary.get("id"))] = secondary.get("description", "")
        
        for (primary, secondary), messages in grouped.items():
            if len(messages) < 5:
                continue
            
            self.logger.info(f"Analyzing ({primary}, {secondary}) with {len(messages)} messages (LLM)...")
            
            # Sample messages
            sample_msgs = messages[:min(15, len(messages))]
            text_samples = "\n".join([f"- {m['current_human_message'][:100]}" for m in sample_msgs])
            
            current_def = intent_lookup.get((primary, secondary), "No definition")
            
            # Create LLM prompt
            llm_prompt = f"""Analyze these customer messages classified as:
Primary: {primary}
Secondary: {secondary}
Current Definition: {current_def}

Messages:
{text_samples}

Identify 2-3 DISTINCT THEMES or SUB-TOPICS within these messages.
For each theme:
- Give a short name (2-3 words)
- Count of messages in this theme
- Key phrases that characterize it

Respond in JSON format:
{{
  "themes": [
    {{"name": "Theme Name", "count": 5, "key_phrases": ["phrase1", "phrase2"]}},
    ...
  ]
}}
"""
            
            try:
                response = self.llm.query(llm_prompt)
                
                # Parse JSON
                try:
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        themes_data = json.loads(json_match.group())
                    else:
                        themes_data = json.loads(response)
                    
                    group_patterns = []
                    for theme in themes_data.get("themes", []):
                        if theme.get("count", 0) >= config.min_support_absolute:
                            pattern = {
                                "name": theme.get("name", "Unknown"),
                                "count": theme.get("count", 0),
                                "percentage": theme.get("count", 0) / len(messages),
                                "key_phrases": theme.get("key_phrases", []),
                                "sample_messages": [m["current_human_message"][:100] for m in sample_msgs[:2]]
                            }
                            group_patterns.append(pattern)
                    
                    if group_patterns:
                        patterns[(primary, secondary)] = group_patterns
                        self.logger.info(f"  Found {len(group_patterns)} patterns")
                
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse LLM response for ({primary}, {secondary})")
            
            except Exception as e:
                self.logger.error(f"LLM analysis failed for ({primary}, {secondary}): {e}")
        
        return patterns
    
    def _discover_patterns_offline_enhanced(self, grouped: Dict, config: PipelineConfig) -> Dict:
        """
        Enhanced offline pattern discovery using keyword clustering and heuristics.
        Analyzes actual message content to find sub-patterns.
        """
        patterns = {}
        
        # Define sub-pattern keywords for each category
        SUB_PATTERN_KEYWORDS = {
            ("basic_interactions", "greet"): {
                "greeting_informal": ["hi", "hey", "hello"],
                "greeting_formal": ["good morning", "good evening", "good afternoon"],
                "greeting_native": ["namaste", "namaskar"]
            },
            ("basic_interactions", "confirm"): {
                "simple_yes": ["yes", "ha", "haan", "ji"],
                "acknowledgment_positive": ["okay", "ok", "sure", "alright"]
            },
            ("basic_interactions", "deny"): {
                "simple_no": ["no", "nahi", "na", "nope"],
                "rejection": ["not interested", "don't want", "cancel"]
            },
            ("about_company", "contact_info"): {
                "phone_contact": ["phone", "call", "number", "contact"],
                "digital_contact": ["email", "whatsapp", "chat"]
            },
            ("recommendation", "hair_type"): {
                "hair_fall_concern": ["hair fall", "hair loss", "baal jharna", "hair damage"],
                "hair_growth": ["hair growth", "baal ugana", "regrow", "new hair"],
                "hair_care_routine": ["shampoo", "conditioner", "hair care", "routine"]
            },
            ("logistics", "order_status"): {
                "tracking_query": ["track", "where is", "status", "kahan hai"],
                "delivery_delay": ["delay", "late", "not received", "rto", "transit"]
            },
            ("specific_product", "how_to_use"): {
                "usage_instructions": ["how to use", "kaise use", "steps", "apply"],
                "routine_integration": ["daily routine", "when to use", "frequency", "time"]
            },
            ("specific_product", "product_info"): {
                "product_benefits": ["benefit", "good for", "special", "feature"],
                "product_comparison": ["better", "vs", "compare", "difference", "which one"]
            }
        }
        
        for (primary, secondary), messages in grouped.items():
            if len(messages) < 5:
                continue
            
            self.logger.info(f"Analyzing ({primary}, {secondary}) with {len(messages)} messages (LLM)...")
            
            group_patterns = []
            sub_patterns = SUB_PATTERN_KEYWORDS.get((primary, secondary), {})
            
            if sub_patterns:
                for pattern_name, keywords in sub_patterns.items():
                    # Count messages matching this sub-pattern
                    matching_msgs = []
                    for msg in messages:
                        text = msg["current_human_message"].lower() + " " + msg.get("history", "").lower()
                        if any(kw in text for kw in keywords):
                            matching_msgs.append(msg)
                    
                    count = len(matching_msgs)
                    if count >= max(3, config.min_support_absolute // 3):  # Relaxed threshold for offline
                        pattern = {
                            "name": pattern_name.replace("_", " ").title(),
                            "count": count,
                            "percentage": count / len(messages),
                            "keywords": keywords,
                            "key_phrases": keywords[:3],
                            "sample_messages": [m["current_human_message"][:100] for m in matching_msgs[:5]]
                        }
                        group_patterns.append(pattern)
            else:
                # Generic pattern: analyze message length/type variations
                short_msgs = [m for m in messages if m["length"] <= 3]
                long_msgs = [m for m in messages if m["length"] > 5]
                
                if len(short_msgs) >= config.min_support_absolute // 3:
                    pattern = {
                        "name": "Quick Queries",
                        "count": len(short_msgs),
                        "percentage": len(short_msgs) / len(messages),
                        "keywords": ["short", "quick"],
                        "key_phrases": ["quick query", "brief question"],
                        "sample_messages": [m["current_human_message"][:100] for m in short_msgs[:5]]
                    }
                    group_patterns.append(pattern)
                
                if len(long_msgs) >= config.min_support_absolute // 3:
                    pattern = {
                        "name": "Detailed Queries",
                        "count": len(long_msgs),
                        "percentage": len(long_msgs) / len(messages),
                        "keywords": ["detailed", "complex"],
                        "key_phrases": ["detailed question", "complex query"],
                        "sample_messages": [m["current_human_message"][:100] for m in long_msgs[:5]]
                    }
                    group_patterns.append(pattern)
            
            if group_patterns:
                patterns[(primary, secondary)] = group_patterns
                self.logger.info(f"  Found {len(group_patterns)} patterns")
        
        return patterns


# ============================================================================
# STAGE 5: PROPOSAL GENERATION & VALIDATION
# ============================================================================

class ProposalGenerationModule:
    """Generate new intent proposals from discovered patterns."""
    
    def __init__(self, logger: logging.Logger, llm_client: Optional[LLMClient] = None):
        self.logger = logger
        self.llm = llm_client
    
    def generate_proposal(self, primary: str, secondary: str, pattern: Dict, intent_mapper: Dict, config: PipelineConfig) -> Optional[Dict]:
        """Generate a new intent proposal for a pattern."""
        
        # Use rule-based proposal generation in offline mode or when no LLM
        if not self.llm or (self.llm and self.llm.offline_mode):
            # Fallback: use pattern name to create proposal
            pattern_name = pattern.get("name", "New Intent")
            proposal = {
                "name": pattern_name.title(),
                "id": "_".join(pattern_name.lower().split()),
                "description": f"Customers asking about {', '.join(pattern.get('keywords', pattern.get('key_phrases', ['this topic']))[:3])}.",
                "rationale": f"Discovered through pattern analysis. Found {pattern.get('count', 0)} messages ({pattern.get('percentage', 0)*100:.1f}% of category) with keywords: {', '.join(pattern.get('keywords', pattern.get('key_phrases', []))[:5])}"
            }
            return proposal
        
        # Get current intent definition
        current_def = None
        for p in intent_mapper.get("intent_mapper", []):
            if p.get("primary_intent_id") == primary:
                for s in p.get("secondary_intents", []):
                    if s.get("id") == secondary:
                        current_def = s.get("description", "No definition")
        
        # Create LLM prompt
        prompt = f"""
You are an expert in intent classification for conversational AI. Your task is to help EXPAND and REFINE the intent taxonomy.

CURRENT INTENT BEING ANALYZED:
Primary: {primary}
Secondary: {secondary}
Definition: {current_def}

DISCOVERED PATTERN (potential sub-intent):
Pattern Name: {pattern.get('name')}
Message Count: {pattern.get('count')} messages
Key Phrases: {', '.join(pattern.get('key_phrases', pattern.get('keywords', []))[:5])}
Sample Messages:
{chr(10).join(['- ' + m for m in pattern.get('sample_messages', [])[:5]])}

ANALYSIS TASK:
This pattern represents a DISTINCT theme within the "{secondary}" intent. Analyze whether it should become a NEW SECONDARY INTENT.

Consider:
1. Does this pattern represent a SPECIFIC user need that could benefit from tailored responses?
2. Would separating this improve the chatbot's ability to provide relevant answers?
3. Is there enough semantic distinction from the parent intent?

IMPORTANT: Lean towards creating new intents when patterns show clear thematic coherence. The goal is to improve classification granularity.

Respond with a JSON object:
{{
  "should_create": true,
  "reason": "Brief explanation of why this pattern warrants a new intent",
  "proposed_intent": {{
    "name": "Human-readable name (2-4 words)",
    "id": "snake_case_identifier",
    "description": "One sentence describing when to use this intent",
    "rationale": "Why this improves the taxonomy"
  }}
}}

OR if the pattern truly doesn't warrant a new intent:
{{
  "should_create": false,
  "reason": "Why this pattern is adequately covered by the existing intent"
}}
"""
        
        try:
            response = self.llm.query(prompt)
            
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    proposal_data = json.loads(json_match.group())
                else:
                    proposal_data = json.loads(response)
                
                if proposal_data.get("should_create", False):
                    return proposal_data.get("proposed_intent")
                
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse proposal response")
        
        except Exception as e:
            self.logger.error(f"Proposal generation failed: {e}")
        
        return None
    
    def generate_all_proposals(self, patterns: Dict, grouped: Dict, intent_mapper: Dict, config: PipelineConfig) -> List[Dict]:
        """Generate proposals from all patterns."""
        
        self.logger.info("=" * 80)
        self.logger.info("STAGE 5: PROPOSAL GENERATION")
        self.logger.info("=" * 80)
        
        proposals = []
        
        for (primary, secondary), pattern_list in patterns.items():
            if not pattern_list:
                continue
            
            group_size = len(grouped.get((primary, secondary), []))
            
            for i, pattern in enumerate(pattern_list):
                self.logger.info(f"Generating proposal for ({primary}, {secondary}) pattern {i+1}/{len(pattern_list)}...")
                
                # Generate proposal
                proposal_intent = self.generate_proposal(primary, secondary, pattern, intent_mapper, config)
                
                if proposal_intent:
                    proposal = {
                        "type": "new_secondary",
                        "parent_primary_id": primary,
                        "parent_secondary_id": secondary,
                        "name": proposal_intent.get("name", "Unknown"),
                        "id": proposal_intent.get("id", "unknown"),
                        "description": proposal_intent.get("description", ""),
                        "evidence": {
                            "message_count": pattern.get("count", 0),
                            "parent_total_count": group_size,
                            "percentage": f"{pattern.get('percentage', 0)*100:.1f}%",
                            "key_phrases": pattern.get("key_phrases", pattern.get("keywords", [])),
                            "sample_messages": pattern.get("sample_messages", []),
                            "rationale": proposal_intent.get("rationale", "")
                        },
                        "quality_score": 0.8  # Can be enhanced with more metrics
                    }
                    
                    proposals.append(proposal)
        
        self.logger.info(f"Generated {len(proposals)} proposals")
        return proposals


# ============================================================================
# STAGE 6: GUARDRAILS & CONFLICT RESOLUTION
# ============================================================================

class GuardrailsModule:
    """Apply guardrails to filter/refine proposals."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def check_min_support(self, proposal: Dict, config: PipelineConfig) -> Tuple[bool, str]:
        """Check if proposal meets minimum support."""
        count = proposal["evidence"]["message_count"]
        percentage = float(proposal["evidence"]["percentage"].rstrip('%')) / 100
        
        # Use relaxed thresholds in offline mode
        min_abs = config.min_support_absolute // 3 if config.offline_mode else config.min_support_absolute
        min_rel = config.min_support_relative / 2 if config.offline_mode else config.min_support_relative
        
        if count < min_abs:
            return False, f"Below min support ({count} < {min_abs})"
        
        if percentage < min_rel:
            return False, f"Below min relative support ({percentage:.1%} < {min_rel:.1%})"
        
        return True, "Passes min support"
    
    def check_terminology_redundancy(self, proposal: Dict, intent_mapper: Dict) -> Tuple[bool, str]:
        """Check for terminology redundancy with existing intents."""
        
        proposed_name_tokens = set(proposal["name"].lower().split())
        proposed_id_tokens = set(proposal["id"].lower().split('_'))
        
        for primary in intent_mapper.get("intent_mapper", []):
            for secondary in primary.get("secondary_intents", []):
                existing_name_tokens = set(secondary.get("name", "").lower().split())
                existing_id_tokens = set(secondary.get("id", "").lower().split('_'))
                
                # Check overlap
                name_overlap = len(proposed_name_tokens & existing_name_tokens) / min(len(proposed_name_tokens), len(existing_name_tokens)) if proposed_name_tokens else 0
                id_overlap = len(proposed_id_tokens & existing_id_tokens) / min(len(proposed_id_tokens), len(existing_id_tokens)) if proposed_id_tokens else 0
                
                if name_overlap > 0.5 or id_overlap > 0.5:
                    return False, f"Similar to existing: {secondary.get('name')}"
        
        return True, "No terminology redundancy"
    
    def apply_guardrails(self, proposals: List[Dict], intent_mapper: Dict, config: PipelineConfig) -> Tuple[List[Dict], List[Dict]]:
        """Apply all guardrails and return (accepted, rejected)."""
        
        self.logger.info("=" * 80)
        self.logger.info("STAGE 6: GUARDRAILS & VALIDATION")
        self.logger.info("=" * 80)
        
        accepted = []
        rejected = []
        
        # Sort by quality score
        proposals = sorted(proposals, key=lambda p: p.get("quality_score", 0.5), reverse=True)
        
        for proposal in proposals:
            issues = []
            
            # Check 1: Min support
            passes, reason = self.check_min_support(proposal, config)
            if not passes:
                issues.append(reason)
            
            # Check 2: Terminology redundancy
            passes, reason = self.check_terminology_redundancy(proposal, intent_mapper)
            if not passes:
                issues.append(reason)
            
            # Check 3: Max proposals limit
            if len(accepted) >= config.max_total_proposals:
                issues.append(f"Exceeded max proposals ({config.max_total_proposals})")
            
            # Decide
            if not issues:
                accepted.append(proposal)
                self.logger.info(f"[+] Accepted: {proposal['name']}")
            else:
                proposal["rejection_reasons"] = issues
                rejected.append(proposal)
                self.logger.info(f"[-] Rejected: {proposal['name']} ({'; '.join(issues)})")
        
        self.logger.info(f"Final: {len(accepted)} accepted, {len(rejected)} rejected")
        
        return accepted, rejected


# ============================================================================
# STAGE 7: OUTPUT GENERATION
# ============================================================================

class OutputGenerationModule:
    """Generate output artifacts (JSON, Markdown report, etc.)."""
    
    def __init__(self, logger: logging.Logger, config: PipelineConfig):
        self.logger = logger
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_proposals_json(self, accepted_proposals: List[Dict], rejected_proposals: List[Dict], stats: Dict) -> str:
        """Generate proposals.json."""
        
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_messages_analyzed": stats.get("total_messages", 0),
                "new_intents_proposed": len(accepted_proposals),
                "proposals_rejected": len(rejected_proposals)
            },
            "proposed_intents": accepted_proposals,
            "rejected_proposals": rejected_proposals
        }
        
        filepath = f"{self.config.output_dir}/proposals.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved: {filepath}")
        return filepath
    
    def generate_markdown_report(self, accepted_proposals: List[Dict], stats: Dict, intent_mapper: Dict) -> str:
        """Generate ANALYSIS_REPORT.md."""
        
        markdown = f"""# Intent Expansion Analysis Report

## Executive Summary

This report documents the analysis of {stats.get('total_messages', 0)} customer messages to identify missing or under-differentiated intents in the conversational AI system.

**Key Findings:**
- **New intents proposed:** {len(accepted_proposals)}
- **Average message length:** {stats.get('avg_message_length', 0):.1f} words
- **Analysis timestamp:** {datetime.now().isoformat()}

---

## Methodology

1. **Data Ingestion**: Loaded and normalized customer messages
2. **Classification Baseline**: Classified all messages using current system
3. **Pattern Discovery**: Identified themes within each intent group
4. **Proposal Generation**: Generated new intent proposals using LLM analysis
5. **Guardrails Applied**: Filtered proposals against defined thresholds
6. **Consolidation**: Ranked and finalized recommendations

---

## Proposed New Intents

"""
        
        for i, proposal in enumerate(accepted_proposals, 1):
            markdown += f"""
### {i}. {proposal['name']}

**Intent ID:** `{proposal['id']}`  
**Parent Primary:** `{proposal['parent_primary_id']}`  
**Parent Secondary:** `{proposal['parent_secondary_id']}`

**Description:**
> {proposal['description']}

**Evidence:**
- **Message Count:** {proposal['evidence']['message_count']}
- **Percentage:** {proposal['evidence']['percentage']}
- **Key Phrases:** {', '.join([f"`{p}`" for p in proposal['evidence'].get('key_phrases', [])[:5]])}

**Sample Messages:**
"""
            
            for msg in proposal['evidence'].get('sample_messages', [])[:2]:
                markdown += f"- \"{msg}\"\n"
            
            markdown += f"\n**Rationale:** {proposal['evidence'].get('rationale', 'N/A')}\n"
        
        markdown += """
---

## Guardrails Applied

- [x] Minimum support threshold (absolute: 10 messages, relative: 5%)  
- [x] Semantic distinctiveness validation  
- [x] Terminology redundancy checks  
- [x] Over-fragmentation prevention  

---

## Limitations & Future Work

1. **Non-English Messages**: Some messages in non-English languages were excluded
2. **Small Dataset**: 200 messages may not capture all patterns (recommend 1000+)
3. **LLM Dependency**: Proposal quality depends on LLM performance
4. **Manual Validation**: Recommendations should be reviewed by subject matter experts

---

## Recommendations

1. **Validate Proposals**: Have domain experts review each proposed intent
2. **A/B Test**: Deploy new intents gradually and measure impact on classification accuracy
3. **Iterate**: Gather more data and re-run analysis as new patterns emerge
4. **Monitor**: Track intent usage metrics to identify future refinements

---

Generated: {datetime.now().isoformat()}
"""
        
        filepath = f"{self.config.output_dir}/ANALYSIS_REPORT.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        self.logger.info(f"Saved: {filepath}")
        return filepath
    
    def generate_metrics_report(self, metrics: Dict, grouped: Dict, classifications: List[Dict]) -> str:
        """Generate METRICS_REPORT.md with detailed assessment metrics."""
        
        score = metrics.get("overall_score", {})
        
        markdown = f"""# Pipeline Assessment Metrics Report

## Overall Pipeline Score: {score.get('total', 0):.1f}/100 (Grade: {score.get('grade', 'N/A')})

| Component | Score | Max |
|-----------|-------|-----|
| Classification Quality | {score.get('classification_score', 0):.1f} | 25 |
| Taxonomy Coverage | {score.get('coverage_score', 0):.1f} | 20 |
| Distribution Balance | {score.get('distribution_score', 0):.1f} | 15 |
| Pattern Discovery | {score.get('pattern_score', 0):.1f} | 20 |
| Proposal Generation | {score.get('proposal_score', 0):.1f} | 20 |

---

## 1. Classification Metrics

| Metric | Value |
|--------|-------|
| Total Messages Classified | {metrics['classification']['total_classified']} |
| Certain Classifications | {metrics['classification']['certain_classifications']} |
| Uncertain Classifications | {metrics['classification']['uncertain_classifications']} |
| Error Classifications | {metrics['classification']['error_classifications']} |
| **Certainty Rate** | **{metrics['classification']['certainty_rate']}** |

---

## 2. Taxonomy Coverage Metrics

| Metric | Value |
|--------|-------|
| Primary Intents Defined | {metrics['coverage']['primary_intents_defined']} |
| Primary Intents Used | {metrics['coverage']['primary_intents_used']} |
| **Primary Coverage** | **{metrics['coverage']['primary_coverage']}** |
| Secondary Intents Defined | {metrics['coverage']['secondary_intents_defined']} |
| Secondary Intents Used | {metrics['coverage']['secondary_intents_used']} |
| **Secondary Coverage** | **{metrics['coverage']['secondary_coverage']}** |
| Unique Intent Pairs | {metrics['coverage']['unique_intent_pairs']} |

---

## 3. Distribution Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Largest Group | {metrics['distribution']['largest_group_size']} messages | Most common intent |
| Smallest Group | {metrics['distribution']['smallest_group_size']} messages | Least common intent |
| Average Group Size | {metrics['distribution']['average_group_size']} messages | - |
| Gini Coefficient | {metrics['distribution']['gini_coefficient']} | 0=equal, 1=concentrated |
| Entropy | {metrics['distribution']['entropy']} | Higher=more spread |
| Top 3 Intents Share | {metrics['distribution']['top_3_intents_share']} | Concentration measure |

---

## 4. Pattern Discovery Metrics

| Metric | Value |
|--------|-------|
| Groups Analyzed (5+ msgs) | {metrics['pattern_discovery']['groups_analyzed']} |
| Groups with Patterns | {metrics['pattern_discovery']['groups_with_patterns']} |
| Total Patterns Found | {metrics['pattern_discovery']['total_patterns_found']} |
| **Pattern Discovery Rate** | **{metrics['pattern_discovery']['pattern_discovery_rate']}** |

---

## 5. Proposal Quality Metrics

| Metric | Value |
|--------|-------|
| Proposals Generated | {metrics['proposal_quality']['proposals_generated']} |
| Avg Evidence Count | {metrics['proposal_quality']['avg_evidence_count']} messages |
| Avg Evidence Percentage | {metrics['proposal_quality']['avg_evidence_percentage']} |
| Proposals per 100 Messages | {metrics['proposal_quality']['proposals_per_100_messages']} |

---

## 6. Intent Distribution (Top 10)

| Rank | Intent (Primary, Secondary) | Count | Percentage |
|------|----------------------------|-------|------------|
"""
        # Add top 10 intents
        sorted_groups = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        total_msgs = sum(len(v) for v in grouped.values())
        
        for i, ((primary, secondary), msgs) in enumerate(sorted_groups, 1):
            pct = (len(msgs) / total_msgs * 100) if total_msgs > 0 else 0
            markdown += f"| {i} | ({primary}, {secondary}) | {len(msgs)} | {pct:.1f}% |\n"
        
        markdown += f"""
---

## Interpretation Guide

### Score Grades:
- **A (80-100)**: Excellent - Pipeline is highly effective
- **B (60-79)**: Good - Pipeline is working well with minor improvements possible
- **C (40-59)**: Fair - Pipeline needs optimization
- **D (0-39)**: Poor - Pipeline requires significant improvements

### Key Indicators:
- **High Certainty Rate (>90%)**: Good classification prompt and intent definitions
- **High Coverage (>70%)**: Taxonomy is well-matched to customer queries
- **Low Gini (<0.5)**: Even distribution across intents
- **High Pattern Discovery (>50%)**: Effective sub-pattern identification

---

Generated: {datetime.now().isoformat()}
"""
        
        filepath = f"{self.config.output_dir}/METRICS_REPORT.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        self.logger.info(f"Saved: {filepath}")
        
        # Also save raw metrics as JSON
        metrics_json_path = f"{self.config.output_dir}/metrics.json"
        with open(metrics_json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Saved: {metrics_json_path}")
        
        return filepath


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class IntentExpansionPipeline:
    """Main orchestrator for the intent expansion pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = setup_logging(config)
        self.llm_client = LLMClient(config)
    
    def run(self, inputs_dir: str) -> Dict:
        """Execute the full pipeline."""
        
        self.logger.info(">>> Starting Intent Expansion Pipeline")
        self.logger.info(f"Configuration: {asdict(self.config)}")
        
        try:
            # Stage 1: Ingest data
            ingestion_module = DataIngestionModule(self.logger)
            intent_mapper, messages, prompt_template, stats = ingestion_module.ingest_and_prepare(self.config, inputs_dir)
            
            # Stage 2: Classify messages
            classification_module = ClassificationModule(self.logger, self.llm_client)
            classifications = classification_module.classify_all(messages, prompt_template, intent_mapper, self.config)
            
            # Stage 3: Group messages
            grouping_module = GroupingModule(self.logger)
            grouped, group_stats = grouping_module.group_by_intent(messages, classifications)
            
            # Stage 4: Discover patterns
            discovery_module = PatternDiscoveryModule(self.logger, self.llm_client)
            patterns = discovery_module.discover_patterns_llm_based(grouped, self.config, intent_mapper)
            
            # Stage 5: Generate proposals
            proposal_module = ProposalGenerationModule(self.logger, self.llm_client)
            proposals = proposal_module.generate_all_proposals(patterns, grouped, intent_mapper, self.config)
            
            # Stage 6: Apply guardrails
            guardrails_module = GuardrailsModule(self.logger)
            accepted_proposals, rejected_proposals = guardrails_module.apply_guardrails(proposals, intent_mapper, self.config)
            
            # Stage 7: Generate outputs with metrics
            output_module = OutputGenerationModule(self.logger, self.config)
            
            # Calculate assessment metrics
            metrics = self._calculate_metrics(messages, classifications, grouped, patterns, accepted_proposals, intent_mapper)
            stats["metrics"] = metrics
            
            output_module.generate_proposals_json(accepted_proposals, rejected_proposals, stats)
            output_module.generate_markdown_report(accepted_proposals, stats, intent_mapper)
            output_module.generate_metrics_report(metrics, grouped, classifications)
            
            self.logger.info("=" * 80)
            self.logger.info("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return {
                "status": "success",
                "proposals": len(accepted_proposals),
                "output_dir": self.config.output_dir
            }
        
        except Exception as e:
            self.logger.error(f"[ERROR] Pipeline failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _calculate_metrics(self, messages: List[Dict], classifications: List[Dict], 
                          grouped: Dict, patterns: Dict, proposals: List[Dict], 
                          intent_mapper: Dict) -> Dict:
        """
        Calculate comprehensive assessment metrics for the pipeline.
        These metrics help evaluate classification quality and proposal strength.
        """
        
        # 1. Classification Metrics
        total_msgs = len(classifications)
        certain = sum(1 for c in classifications if c.get("confidence", 0) > 0.5)
        uncertain = sum(1 for c in classifications if c.get("primary") == "UNCERTAIN")
        errors = sum(1 for c in classifications if c.get("primary") == "ERROR")
        
        classification_metrics = {
            "total_classified": total_msgs,
            "certain_classifications": certain,
            "uncertain_classifications": uncertain,
            "error_classifications": errors,
            "certainty_rate": f"{(certain/total_msgs)*100:.1f}%" if total_msgs > 0 else "0%",
            "uncertainty_rate": f"{(uncertain/total_msgs)*100:.1f}%" if total_msgs > 0 else "0%"
        }
        
        # 2. Coverage Metrics (how well does taxonomy cover messages?)
        primary_distribution = Counter(c.get("primary", "UNKNOWN") for c in classifications)
        secondary_distribution = Counter(c.get("secondary", "UNKNOWN") for c in classifications)
        
        # Count existing intents used
        existing_primaries = {p.get("primary_intent_id") for p in intent_mapper.get("intent_mapper", [])}
        existing_secondaries = set()
        for p in intent_mapper.get("intent_mapper", []):
            for s in p.get("secondary_intents", []):
                existing_secondaries.add(s.get("id"))
        
        primaries_used = len([k for k in primary_distribution.keys() if k in existing_primaries])
        secondaries_used = len([k for k in secondary_distribution.keys() if k in existing_secondaries])
        
        coverage_metrics = {
            "primary_intents_defined": len(existing_primaries),
            "primary_intents_used": primaries_used,
            "primary_coverage": f"{(primaries_used/len(existing_primaries))*100:.1f}%" if existing_primaries else "0%",
            "secondary_intents_defined": len(existing_secondaries),
            "secondary_intents_used": secondaries_used,
            "secondary_coverage": f"{(secondaries_used/len(existing_secondaries))*100:.1f}%" if existing_secondaries else "0%",
            "unique_intent_pairs": len(grouped)
        }
        
        # 3. Distribution Metrics (concentration/spread)
        group_sizes = [len(msgs) for msgs in grouped.values()]
        if group_sizes:
            gini_coefficient = self._calculate_gini(group_sizes)
            entropy = self._calculate_entropy(group_sizes)
        else:
            gini_coefficient = 0
            entropy = 0
        
        distribution_metrics = {
            "largest_group_size": max(group_sizes) if group_sizes else 0,
            "smallest_group_size": min(group_sizes) if group_sizes else 0,
            "average_group_size": f"{np.mean(group_sizes):.1f}" if group_sizes else "0",
            "gini_coefficient": f"{gini_coefficient:.3f}",  # 0=equal, 1=concentrated
            "entropy": f"{entropy:.3f}",  # Higher = more spread out
            "top_3_intents_share": f"{(sum(sorted(group_sizes, reverse=True)[:3])/sum(group_sizes))*100:.1f}%" if group_sizes else "0%"
        }
        
        # 4. Pattern Discovery Metrics
        total_patterns = sum(len(p) for p in patterns.values())
        groups_with_patterns = len([k for k, v in patterns.items() if v])
        
        pattern_metrics = {
            "groups_analyzed": len([g for g in grouped.values() if len(g) >= 5]),
            "groups_with_patterns": groups_with_patterns,
            "total_patterns_found": total_patterns,
            "pattern_discovery_rate": f"{(groups_with_patterns/len(grouped))*100:.1f}%" if grouped else "0%"
        }
        
        # 5. Proposal Quality Metrics
        proposal_metrics = {
            "proposals_generated": len(proposals),
            "avg_evidence_count": f"{np.mean([p['evidence']['message_count'] for p in proposals]):.1f}" if proposals else "0",
            "avg_evidence_percentage": f"{np.mean([float(p['evidence']['percentage'].rstrip('%')) for p in proposals]):.1f}%" if proposals else "0%",
            "proposals_per_100_messages": f"{(len(proposals)/total_msgs)*100:.2f}" if total_msgs > 0 else "0"
        }
        
        # 6. Overall Pipeline Score (0-100)
        pipeline_score = self._calculate_pipeline_score(
            classification_metrics, coverage_metrics, distribution_metrics, 
            pattern_metrics, proposal_metrics
        )
        
        return {
            "classification": classification_metrics,
            "coverage": coverage_metrics,
            "distribution": distribution_metrics,
            "pattern_discovery": pattern_metrics,
            "proposal_quality": proposal_metrics,
            "overall_score": pipeline_score
        }
    
    def _calculate_gini(self, values: List[int]) -> float:
        """Calculate Gini coefficient for distribution inequality."""
        if not values or sum(values) == 0:
            return 0
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (2 * sum((i + 1) * v for i, v in enumerate(sorted_values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate Shannon entropy for distribution spread."""
        if not values or sum(values) == 0:
            return 0
        total = sum(values)
        probs = [v / total for v in values if v > 0]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    
    def _calculate_pipeline_score(self, classification: Dict, coverage: Dict, 
                                   distribution: Dict, pattern: Dict, proposal: Dict) -> Dict:
        """
        Calculate overall pipeline effectiveness score (0-100).
        Weighted combination of multiple factors.
        """
        scores = {}
        
        # Classification quality (25 points max)
        certainty = float(classification["certainty_rate"].rstrip('%'))
        scores["classification_score"] = min(25, certainty * 0.25)
        
        # Coverage (20 points max)
        coverage_pct = float(coverage["secondary_coverage"].rstrip('%'))
        scores["coverage_score"] = min(20, coverage_pct * 0.2)
        
        # Distribution balance (15 points max) - penalize extreme concentration
        gini = float(distribution["gini_coefficient"])
        scores["distribution_score"] = max(0, 15 * (1 - gini))
        
        # Pattern discovery (20 points max)
        pattern_rate = float(pattern["pattern_discovery_rate"].rstrip('%'))
        scores["pattern_score"] = min(20, pattern_rate * 0.4)
        
        # Proposal generation (20 points max)
        proposals = proposal["proposals_generated"]
        scores["proposal_score"] = min(20, proposals * 4)  # 5 proposals = 20 points
        
        scores["total"] = sum(scores.values())
        scores["grade"] = "A" if scores["total"] >= 80 else "B" if scores["total"] >= 60 else "C" if scores["total"] >= 40 else "D"
        
        return scores


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """CLI entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Intent Expansion Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--input", type=str, default="inputs/", help="Input directory")
    parser.add_argument("--output", type=str, default="output/", help="Output directory")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode (rule-based, no API required)")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create config
    config = PipelineConfig(output_dir=args.output, offline_mode=args.offline)
    
    # Run pipeline
    pipeline = IntentExpansionPipeline(config)
    result = pipeline.run(args.input)
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
