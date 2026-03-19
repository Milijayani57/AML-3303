---
title: "Part 3 — LDA Topic Modeling"
description: "Unsupervised topic discovery on the Reddit Climate Change dataset using Latent Dirichlet Allocation"
course: AML-3203
group: Group 1
input: reddit_preprocessed.csv
output: reddit_lda_labeled.csv
tags: [lda, topic-modeling, nlp, gensim, reddit, climate-change]
---

# Part 3 — LDA Topic Modeling

---

## 1. Objective and Justification

The goal of this stage was to perform **unsupervised topic modeling** on the Reddit Climate Change dataset to discover hidden thematic structures without any manual labeling.

### Text Source Selection

The `title` column was selected as the primary text source. The reason is straightforward — the data showed that `selftext` (post body) is missing in **72.4% of rows**, while `title` is **100% present** across all posts. Using `title` ensures every post contributes to the model, preventing a massive loss of data.

| Column | Availability | Decision |
|---|---|---|
| `title` | 100% present across all 620,908 posts | ✅ Selected |
| `selftext` | Missing in 72.4% of rows | ❌ Rejected |

### Why LDA?

LDA is **unsupervised** — it requires no labeled data. It discovers topics purely from word co-occurrence patterns, which is exactly what is needed when the thematic structure of the corpus is unknown in advance.

---

## 2. Methodology and Pipeline

The analysis followed a sequential NLP pipeline to transform raw Reddit titles into mathematical topic representations.

### Step 1 — Preprocessing

The `title` text was cleaned to remove URLs, Reddit markdown (`>`, `**`), HTML entities, punctuation, and extra whitespace. Text was then lowercased.

### Step 2 — Tokenization and Lemmatization

Cleaned text was split into tokens, stopwords were removed using the NLTK English stopword list, and any token shorter than 2 characters was dropped. Lemmatization was then applied using **spaCy** with the POS tagger active — reducing inflected forms to their root dictionary form so that variants like `"emissions"` and `"emission"` are treated as the same word.

Lemmatisation statistics across 40,459 posts:

| Metric | Value |
|---|---|
| Total lemmas | 711,114 |
| Average lemmas per post | 17.6 |
| Median lemmas per post | 9 |
| Posts with 0 lemmas | 16 |

### Step 3 — Vectorization (Dictionary and Bag-of-Words)

The token lists were converted into the two structures gensim's LDA requires: a **Dictionary** (token → integer ID) and a **Bag-of-Words corpus** (each document as `(token_id, count)` pairs).

Before building these, the vocabulary was filtered to remove noise:

| Parameter | Value | Rationale |
|---|---|---|
| `no_below` | 20 | Must appear in ≥ 20 documents — removes typos and one-off terms |
| `no_above` | 0.50 | Must not appear in > 50% of documents — removes near-universal words |
| `keep_n` | 50,000 | Hard ceiling on retained vocabulary |

Filtering compressed the vocabulary from **24,122 raw tokens → 3,513 tokens**. Average tokens per document after filtering: **10.9**.

> 478 documents became empty after filtering and were assigned topic `-1`. They remain in the DataFrame but contribute nothing to training.

### Step 4 — LDA Training

The final model was trained using `LdaMulticore` with K = 10, selected via a coherence sweep:

| Parameter | Value | Rationale |
|---|---|---|
| `num_topics` | 10 | Best K from coherence tuning |
| `passes` | 20 | More passes → better convergence |
| `alpha` | `'asymmetric'` | Reflects unequal topic prevalence on Reddit |
| `eta` | `'auto'` | Model learns word-distribution prior from data |
| `random_state` | 42 | Reproducibility |

---

## 3. Analysis of Model Outputs

### 3.1 Coherence Score

The model's quality was validated using **C_v coherence**, which measures the semantic similarity of the top words within each topic based on co-occurrence patterns. The final model scored **0.5498** — a solid result for short social media text, where scores above 0.50 are considered acceptable.

### 3.2 Top Words per Topic

After training, the top 12 words for each topic were inspected:

| Topic | Top 12 Words |
|---|---|
| 0 | health, fight, threat, covid, policy, pandemic, solve, leader, bitcoin, act, war, global |
| 1 | people, think, like, don, year, time, world, bad, go, thing, know, believe |
| 2 | new, gas, fight, oil, plan, government, action, biden, city, emission, expert, ban |
| 3 | survey, research, study, project, perception, thank, question, court, student, response, fight, school |
| 4 | news, want, good, stop, help, work, tell, need, video, save, big, fight |
| 5 | impact, european, food, potential, technology, innovation, green, consumer, increase, deal, come, aviation |
| 6 | study, report, weather, year, new, flood, extreme, say, world, find, heat, disaster |
| 7 | area, high, fire, winter, sea, risk, level, region, wildfire, rise, city, california |
| 8 | energy, fuel, power, fossil, need, nuclear, combat, technology, solution, use, company, clean |
| 9 | global, warming, carbon, emission, plant, co, earth, help, tree, ocean, reduce, scientist |

### 3.3 Word Clouds per Topic

Each topic was visualised as a word cloud, with word size proportional to its probability weight within that topic.

 <img width="869" height="602" alt="image" src="https://github.com/user-attachments/assets/51c9157c-9016-4551-b825-d4459af25fc2" />


After reviewing these word clouds, the following human-readable labels were assigned:

| Topic | Label |
|---|---|
| 0 | Climate Policy & Health |
| 1 | Public Opinion & Debate |
| 2 | Government & Oil Policy |
| 3 | Research & Education |
| 4 | Climate Action & News |
| 5 | Economic & Food Impact |
| 6 | Climate Science & Weather |
| 7 | Extreme Weather & Wildfires |
| 8 | Energy & Fossil Fuels |
| 9 | Carbon & Global Emissions |

### 3.4 Word Clouds by Engagement Level

Posts were split at the **95th percentile (high engagement)** and **5th percentile (low engagement)** of score. A word cloud was generated from each group's token frequencies to show how language differs between viral posts and ignored ones.

<img width="1039" height="672" alt="image" src="https://github.com/user-attachments/assets/0c39dda5-1889-4b19-a7b4-1feca97aaf24" />

High-engagement posts are dominated by solution-oriented and science-grounded language. Low-engagement posts show more generic or inflammatory terms.

### 3.5 Topic Assignment

Every post was assigned a `dominant_topic` — the topic with the highest probability in its topic distribution, identified using `lda_final.get_document_topics()`. Posts with empty BoW vectors received `-1`.

Three columns were added to the DataFrame:

| Column | Description |
|---|---|
| `dominant_topic` | Topic ID (0–9) with the highest probability |
| `topic_label` | Human-readable label |
| `topic_prob` | Probability of the dominant topic |

Out of 40,459 total posts, **39,981 received a valid topic assignment** (`dominant_topic >= 0`).

### 3.6 Topic Distribution

The percentage share of each topic shows which themes dominate climate change conversation on Reddit.

<img width="1349" height="410" alt="image" src="https://github.com/user-attachments/assets/109d275e-2218-48dd-a4e6-33d448ac0d05" />


| Topic ID | Label | Posts | % of Total | Avg Score |
|---|---|---|---|---|
| 1 | Public Opinion & Debate | 10,480 | 25.9% | 88.7 |
| 0 | Climate Policy & Health | 6,082 | 15.0% | 131.9 |
| 2 | Government & Oil Policy | 5,060 | 12.5% | 62.9 |
| 6 | Climate Science & Weather | 3,771 | 9.3% | 66.2 |
| 4 | Climate Action & News | 3,238 | 8.0% | 63.0 |
| 9 | Carbon & Global Emissions | 2,837 | 7.0% | 143.0 |
| 3 | Research & Education | 2,709 | 6.7% | 104.2 |
| 5 | Economic & Food Impact | 2,283 | 5.6% | 28.5 |
| 8 | Energy & Fossil Fuels | 2,215 | 5.5% | 150.7 |
| 7 | Extreme Weather & Wildfires | 1,306 | 3.2% | 57.4 |

**Public Opinion & Debate** is the largest topic at 25.9% — confirming Reddit is a discussion-first platform. **Energy & Fossil Fuels** has the highest average score (150.7) despite being among the smallest topics, meaning posts about energy solutions generate the most engagement. **Economic & Food Impact** scores the lowest (28.5), showing that socioeconomic framing of climate change resonates least with this audience.

---

## 4. Visualizations and Handoffs

### 4.1 Topic Trends Over Time

Topic share was calculated per year to track how climate discourse evolved from 2021 to 2022.

<img width="1514" height="516" alt="image" src="https://github.com/user-attachments/assets/303747f1-f497-4afe-9f3f-2ef8fb0ee6a6" />


| Topic Label | 2021 (%) | 2022 (%) | Change |
|---|---|---|---|
| Public Opinion & Debate | 31.1 | 26.0 | ↓ 5.1 pp |
| Climate Policy & Health | 12.5 | 15.4 | ↑ 2.9 pp |
| Research & Education | 3.8 | 6.9 | ↑ 3.1 pp |
| Energy & Fossil Fuels | 4.7 | 5.6 | ↑ 0.9 pp |
| Carbon & Global Emissions | 6.2 | 7.1 | ↑ 0.9 pp |
| Government & Oil Policy | 13.0 | 12.6 | ↓ 0.4 pp |
| Climate Science & Weather | 10.2 | 9.4 | ↓ 0.8 pp |
| Climate Action & News | 9.4 | 8.0 | ↓ 1.4 pp |
| Economic & Food Impact | 5.4 | 5.7 | ↑ 0.3 pp |
| Extreme Weather & Wildfires | 3.6 | 3.2 | ↓ 0.4 pp |

General debate dropped sharply (↓5.1 pp) while Research & Education nearly doubled (↑3.1 pp), consistent with the IPCC AR6 report driving scientific literacy content into 2022. Energy & Fossil Fuels grew in line with the Ukraine-triggered energy security debate.

### 4.2 Inter-Topic Distance Map (pyLDAvis)

An interactive visualization was saved as `lda_visualization.html`, rendering all 10 topics as bubbles in a t-SNE 2D space.

<img width="1040" height="601" alt="image" src="https://github.com/user-attachments/assets/92314ac6-e49a-479b-b15b-e398523138b1" />


> Use **λ = 0.6** on the slider for the most interpretable word ranking. Well-separated bubbles confirm the 10 topics are semantically distinct.

### 4.3 Machine Learning Handoff

The labeled dataset was saved as `reddit_lda_labeled.csv` to serve as the input for the LSTM classifier in the next phase.

| File | Description |
|---|---|
| `reddit_lda_labeled.csv` | 40,459 rows × 11 columns with topic labels |
| `lda_model/` | Serialized gensim model, reloadable without retraining |
| `lda_visualization.html` | Standalone interactive visualization |

---

## Summary

| Metric | Value |
|---|---|
| Total posts analyzed | 40,459 |
| Topics discovered | 10 |
| Final C_v coherence | 0.5498 |
| Posts with valid topic | 39,981 |
| Largest topic | Public Opinion & Debate — 25.9% |
| Highest engagement topic | Energy & Fossil Fuels — avg score 150.7 |
| Lowest engagement topic | Economic & Food Impact — avg score 28.5 |
