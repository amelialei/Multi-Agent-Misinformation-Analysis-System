# Multi-Agent Misinformation Analysis System

## Overview
This project addresses the challenge of identifying misinformation in news content through a multi-agent LLM architecture. Rather than relying on a single verdict, the system evaluates articles across four factuality dimensions including **Frequency Heuristic**, **Sensationalism**, **Malicious Account**, and **Naive Realism**. Each of these factors are analyzed by a a respective specialized AI agent.

The system uses the [LIAR-PLUS dataset](https://github.com/Tariq60/LIAR-PLUS) (augmented with scraped PolitiFact data) to ground each agent's analytical framework, and routes all results through an orchestrator and merger agent to produce a final truthfulness label with per-factor reasoning.

## Agent Architecture

The system consists of five specialized agents coordinated by an orchestrator and merger:

1. **Claim Extraction Agent** — Extracts 3–7 verifiable factual claims from an article, filters opinions and speculation, and uses the Claim Verification Agent as a tool with Google Search integration for real-time fact-checking. Instead of traditional RAG, our system relies entirely on live search for external knowledge retrieval.
2. **Frequency Heuristic Agent** — Analyzes repetition patterns, buzzword usage, and linguistic cues that artificially enhance perceived "truthiness."
3. **Sensationalism Agent** — Detects emotionally charged, exaggerated, or dramatic rhetoric.
4. **Malicious Account Agent** — Identifies linguistic markers associated with bot-like or inauthentic messaging behavior.
5. **Naive Realism Agent** — Measures absolutist, polarized, or dismissive language that presents opinion as fact.

**Orchestrator** — A `SequentialAgent` that calls each specialized sub-agent and coordinates the full analysis pipeline.

**Merger Agent** — Aggregates outputs from all four factuality agents to produce a final truthfulness label with a synthesized explanation.

### Verification Statuses - Claim Extraction Agent

| Status | Meaning |
|---|---|
| `SUPPORTED` | Multiple credible sources confirm the claim |
| `REFUTED` | Credible sources contradict the claim |
| `PARTIALLY_SUPPORTED` | Mixed or incomplete evidence |
| `UNVERIFIABLE` | Insufficient information available |
| `CONFLICTING` | Sources disagree |

### Prompt Variants
Each factuality factor agent supports seven prompt variants for systematic experimentation:

`base` · `cot` (Chain-of-Thought) · `fcot` / `fcot2` / `fcot3` (Fractal CoT) · `function_calling` · `icl` (In-Context Learning)

## Repository Structure
```text
DSC180A-Q2Project/
├── data/
│   ├── article.txt                   # Article text for ingestion
│   ├── ground_truth.csv              # Hand-labeled articles with factuality factor scores
│   ├── politifact.csv                # Manually scraped data from Politifact.org to augment LIAR-PLUS
│   ├── train_set.csv                 # Training set with scraped data
│   ├── train2.tsv                    # Original LiarPLUS train set
│   ├── val_set.csv                   # Validation set with scraped data
│   ├── val2.tsv                      # Original LiarPLUS validation set
│   ├── test_set.csv                  # Test set with scraped data
│   ├── test2.tsv                     # Original LiarPLUS test set
│   ├── base_results.csv              # Base prompt experiment results
│   ├── cot_results.csv               # Chain-of-thought prompt results
│   ├── fcot1_results.csv             # Fractal CoT (v1) results
│   ├── fcot2_results.csv             # Fractal CoT (v2) results
│   ├── fcot_gemini3_results.csv      # Fractal CoT results using Gemini 3
│   ├── function_calling_results.csv  # Tool-based prompt results
│   └── icl_results.csv               # In-context learning prompt results
│
├── notebooks/
│   ├── agent_experiments.ipynb       # Accuracy evaluation of agent prompt variants
│   ├── eda_visualization.ipynb       # Exploratory visualizations for LIAR-PLUS dataset
│   ├── metrics.ipynb                 # Accuracy scores for predictive and LLM models
│   ├── model_accuracy.ipynb          # Performance metrics for baseline predictive models
│   ├── prompt_results.ipynb          # Prompting experiment visualizations
│   ├── prompting.ipynb               # 20 incremental prompts refining the model
│   └── scraped_data.ipynb            # Additional scraped data from Politifact added to LiarPLUS
│
├── ff_agents/                        # Multi-agent factuality analysis system
│   ├── claim_extraction_agent/       # Claim extraction (single instruction, no prompts)
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── frequency_heuristic_agent/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── prompts/                  # Prompt variants used for experiments
│   │       ├── base.txt
│   │       ├── cot.txt
│   │       ├── fcot.txt
│   │       ├── fcot2.txt
│   │       ├── fcot3.txt
│   │       ├── function_calling.txt
│   │       └── icl.txt
│   ├── sensationalism_agent/         # Same structure as frequency_heuristic_agent
│   ├── malicious_account_agent/      # Same structure as frequency_heuristic_agent
│   ├── naive_realism_agent/          # Same structure as frequency_heuristic_agent
│   ├── orchestrator/                 # Controls agent execution flow
│   │   ├── __init__.py
│   │   └── agent.py
│   └── merger_agent/                 # Aggregates agent outputs
│       ├── __init__.py
│       └── agent.py
│
├── src/                              # Core project source code
│   ├── __init__.py
│   ├── articles.py                   # Article ingestion and preprocessing
│   ├── predictive_models.py          # ML prediction pipeline
│   ├── script.py                     # Main script to run full ML model pipeline
│   └── config.json                   # Config settings for models and pipeline
│
├── webapp/                           # Flask-based UI integrated with agent pipeline
│   ├── app.py                        # Flask entrypoint
│   ├── prompts/
│   │   ├── base.txt                      # Base prompt for LLM
│   │   ├── chain_of_thought.txt          # Chain-of-thought LLM prompt
│   │   └── fractal_chain_of_thought.txt  # Fractal chain-of-thought LLM prompt
│   ├── results/
│   │   ├── base_outputs.csv              # Outputs from base prompt runs
│   │   ├── cot_outputs.csv               # Outputs from chain-of-thought prompt runs
│   │   └── fcot_outputs.csv              # Outputs from fractal CoT prompt runs
│   ├── static/
│   │   └── style.css                 # CSS styling for UI
│   └── templates/
│       └── index.html                # Main UI page
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

This project uses the **LIAR-PLUS** dataset, an extended version of the original LIAR dataset, augmented with recent data scraped from PolitiFact. It includes labeled political statements with metadata such as subjects, speakers, party affiliations, and justifications.

| Split | File | Description |
|-------|------|-------------|
| Train | `train_set.csv` | Used to train all factuality models |
| Validation | `val_set.csv` | Used for tuning and intermediate evaluation |
| Test | `test_set.csv` | Used for final evaluation and analysis |

## Installation

### Clone the repository
```bash
git clone https://github.com/JacquelynGarcia/DSC180A-Q1Project.git
cd Multi-Agent-Misinformation-Analysis-System
```

### Create a virtual environment
```bash
python -m venv venv

source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### Create a .env file
Create a `.env` file in the `webapp` folder and paste your API key from [Google AI Studio](https://aistudio.google.com):
```env
GEMINI_API_KEY=YOUR_API_KEY
```

### Create a .env file in ff_agents/
Create a `.env` file in the `ff_agents` folder with the following variables:
```env
GOOGLE_GENAI_USE_VERTEXAI=0
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```
Get your API key from [Google AI Studio](https://aistudio.google.com).

### Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Agent Pipeline

### Via ADK Web Interface
```bash
adk web ff_agents
```
1. Select the **orchestrator** from the top-left dropdown menu
2. Paste an article into the chat to analyze
3. Individual agents can also be selected and tested independently from the dropdown

### Changing Prompt Variants
To switch prompt variants for any factuality agent, edit the corresponding `agent.py` and update the prompt loading line:
```python
prompt = load_prompt(_AGENT_DIR, "file_name.txt")
```
Available files: `base.txt`, `cot.txt`, `fcot.txt`, `fcot2.txt`, `fcot3.txt`, `function_calling.txt`, `icl.txt`

## Running the Web Application

The Flask UI is integrated with the agent pipeline and supports end-to-end article verification.

**Features:**
- Paste an article link or raw text
- View scores for each factuality factor
- View agent reasoning and confidence percentages
- Clear button to reset and analyze a new article
- All analyses automatically saved to `results/data_outputs.csv`

### Start the app
```bash
cd webapp
python app.py
```

### Open in your browser
```
http://127.0.0.1:5000
```

## Example Output

**This article is: True**

> The article is factually accurate, as all major claims were fully supported by external evidence. While it utilizes highly sensationalized language and repetitive 'monarchical' framing, it provides traceable sources and includes counter-perspectives. The high degree of factual consistency offsets the heavily biased tone, resulting in a truthful rating.

| Factuality Factor | Score | Confidence | Reasoning |
|---|---|---|---|
| Frequency Heuristic | 1 | Frequent repetition of 'king' and 'monarchical' framing; relies on popularity signals like 'millions of protesters' | 85% |
| Malicious Account | 0 | Cites traceable sources (CNN, NBC, AP, Reuters); lacks indicators of inauthentic behavior | 85% |
| Sensationalism | 2 | Highly charged framing: 'wannabe autocrat,' 'regal whims,' prioritizes emotional escalation | 85% |
| Naive Realism | 1 | Interpretive and critical language; dismisses alternative viewpoints to reinforce central narrative | 85% |

## Requirements

```
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
xgboost==2.1.1
textblob==0.18.0
scipy==1.14.1
wordcloud==1.9.3
feedparser==6.0.10
sentence-transformers==2.7.0
transformers==4.44.2
chromadb==0.4.24
beautifulsoup4==4.12.2
requests==2.31.0
spacy==3.7.2
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
matplotlib==3.7.2
seaborn==0.12.2
lxml_html_clean==0.1.0
google-genai
python-dotenv
flask==3.0.3
torch==2.3.1
google-adk
google-adk[a2a]
```

> **macOS only:** If you encounter an OpenMP error when running XGBoost, install the OpenMP runtime via Homebrew:
> ```bash
> brew install libomp
> ```

## Acknowledgment

This project was developed as part of the HDSI Capstone program.  
For additional information and related work, please visit: https://alternusvera.com

## License
This project is part of the DSC180B Capstone course at UC San Diego.