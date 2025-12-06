# Intent Expansion Analysis Report

## Executive Summary

This report documents the analysis of 200 customer messages to identify missing or under-differentiated intents in the conversational AI system.

**Key Findings:**
- **New intents proposed:** 4
- **Average message length:** 6.5 words
- **Analysis timestamp:** 2025-12-03T16:57:51.653531

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


### 1. Order Tracking

**Intent ID:** `order_tracking`  
**Parent Primary:** `logistics`  
**Parent Secondary:** `order_status`

**Description:**
> When customers specifically want to track their order or need tracking information/instructions.

**Evidence:**
- **Message Count:** 9
- **Percentage:** 20.9%
- **Key Phrases:** `how to track order`, `track order`, `tracking`

**Sample Messages:**
- "how to track order?"
- "mera order confirm ho gaya hai"

**Rationale:** Separating tracking requests from general status inquiries allows for specialized responses - tracking requests need specific tracking links, order numbers, or tracking instructions, while status inquiries may need broader order information or explanations about delays/processing.

### 2. Usage Instructions

**Intent ID:** `usage_instructions`  
**Parent Primary:** `specific_product`  
**Parent Secondary:** `product_info`

**Description:**
> When customers specifically ask for instructions on how to use, apply, or operate a product

**Evidence:**
- **Message Count:** 8
- **Percentage:** 19.5%
- **Key Phrases:** `how to use`, `how to used`, `how will i apply`, `how to apply`, `kaise use kare`

**Sample Messages:**
- "others details"
- "before use serum oil can be applied"

**Rationale:** Separating usage queries from general product info allows for targeted responses with step-by-step instructions, application methods, and practical guidance rather than broad product details

### 3. Product Benefits Inquiry

**Intent ID:** `product_benefits_inquiry`  
**Parent Primary:** `specific_product`  
**Parent Secondary:** `product_speciality_uc`

**Description:**
> When customer specifically asks about the benefits, results, or what makes a product special/effective

**Evidence:**
- **Message Count:** 11
- **Percentage:** 45.8%
- **Key Phrases:** `what's special about it?`, `specialty`, `benefits`, `results`

**Sample Messages:**
- "what's special about it? ultra smoothing shampoo for smooth & shiny hair- 250ml"
- "seee results"

**Rationale:** Benefits-focused queries require responses emphasizing outcomes and value propositions, while general specialty questions may cover features, uniqueness, or technical specifications - this distinction enables more targeted and persuasive responses

### 4. Hair Care Recommendations

**Intent ID:** `hair_care_recommendations`  
**Parent Primary:** `recommendation`  
**Parent Secondary:** `seeking_solution`

**Description:**
> When customer seeks product recommendations specifically for hair-related concerns including growth, loss, scalp health, or hair care routines

**Evidence:**
- **Message Count:** 6
- **Percentage:** 60.0%
- **Key Phrases:** `hair growth`, `shampoo`, `hair oil`, `baldness`, `baby hair growth`

**Sample Messages:**
- "which facewash is good for brighter face"
- "my lips like in two tone give me your best product"

**Rationale:** Hair care has unique product categories (shampoos, oils, treatments), different ingredient considerations, and specialized concerns that benefit from targeted responses separate from general skincare recommendations

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
