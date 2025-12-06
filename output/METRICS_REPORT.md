# Pipeline Assessment Metrics Report

## Overall Pipeline Score: 70.6/100 (Grade: B)

| Component | Score | Max |
|-----------|-------|-----|
| Classification Quality | 25.0 | 25 |
| Taxonomy Coverage | 14.9 | 20 |
| Distribution Balance | 5.8 | 15 |
| Pattern Discovery | 8.9 | 20 |
| Proposal Generation | 16.0 | 20 |

---

## 1. Classification Metrics

| Metric | Value |
|--------|-------|
| Total Messages Classified | 200 |
| Certain Classifications | 200 |
| Uncertain Classifications | 0 |
| Error Classifications | 0 |
| **Certainty Rate** | **100.0%** |

---

## 2. Taxonomy Coverage Metrics

| Metric | Value |
|--------|-------|
| Primary Intents Defined | 5 |
| Primary Intents Used | 5 |
| **Primary Coverage** | **100.0%** |
| Secondary Intents Defined | 35 |
| Secondary Intents Used | 26 |
| **Secondary Coverage** | **74.3%** |
| Unique Intent Pairs | 27 |

---

## 3. Distribution Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Largest Group | 43 messages | Most common intent |
| Smallest Group | 1 messages | Least common intent |
| Average Group Size | 7.4 messages | - |
| Gini Coefficient | 0.611 | 0=equal, 1=concentrated |
| Entropy | 3.728 | Higher=more spread |
| Top 3 Intents Share | 54.0% | Concentration measure |

---

## 4. Pattern Discovery Metrics

| Metric | Value |
|--------|-------|
| Groups Analyzed (5+ msgs) | 9 |
| Groups with Patterns | 6 |
| Total Patterns Found | 6 |
| **Pattern Discovery Rate** | **22.2%** |

---

## 5. Proposal Quality Metrics

| Metric | Value |
|--------|-------|
| Proposals Generated | 4 |
| Avg Evidence Count | 8.5 messages |
| Avg Evidence Percentage | 36.5% |
| Proposals per 100 Messages | 2.00 |

---

## 6. Intent Distribution (Top 10)

| Rank | Intent (Primary, Secondary) | Count | Percentage |
|------|----------------------------|-------|------------|
| 1 | (logistics, order_status) | 43 | 21.5% |
| 2 | (specific_product, product_info) | 41 | 20.5% |
| 3 | (specific_product, product_speciality_uc) | 24 | 12.0% |
| 4 | (basic_interactions, acknowledgment) | 13 | 6.5% |
| 5 | (basic_interactions, greetings) | 11 | 5.5% |
| 6 | (recommendation, seeking_solution) | 10 | 5.0% |
| 7 | (specific_product, effectiveness) | 9 | 4.5% |
| 8 | (recommendation, customer_information) | 5 | 2.5% |
| 9 | (logistics, complaint) | 5 | 2.5% |
| 10 | (specific_product, all_product) | 4 | 2.0% |

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

Generated: 2025-12-03T16:57:51.654658
