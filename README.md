# DSC180A Capstone Project - Multi-Agent Misinformation Analysis System

## Overview
This project addresses the challenges of identifying misinformation in news content by combining traditional ML models with a multi-agent LLM architecture. Instead of relying on one source of truth, our system evaluates multiple factuality dimensions including frequency heuristic, sensationalism, malicious account, and naive realism.

The project has evolved from a standalone ML pipeline with into an agent-based system designed to support claim analysis, verification, and explainability. The current phase focuses on agent integration, with plans to connect the system to the interactive interface next.

We used the LIAR-PLUS dataset to train separate models for:

- Frequency Heuristics - detecting repetition, buzzwords, and linguistic patterns that artificially enhance “truthiness.”
- Sensationalism - identifying emotionally charged, exaggerated, or dramatic rhetoric.
- Malicious Account – detecting linguistic markers commonly associated with bot-like or spammy messaging.
- Naive Realism – capturing absolutist, polarized, or dismissive language that reflects cognitive bias

Each model captures a unique dimension of factuality, contributing to a broader framework for automated fact-checking.

## Recent Updates - Agent Integration (Q2)
We've integrated a multi-agent verification system that enhances our fact-checking pipeline with AI-powered agents:

### Agent Architecture
The system consists of five specialized agents.
1. Claim Extraction Agent: Extracts verifiiable factual claims from articles
- Uses Claim Verification Agent as a tool with Google Search integraton
- No traditional RAG - relies on Google Search for external knowledge retrieval
2. Frequency Heuristic Agent: Analyzes repetition patterns and buzzword usage
3. Sensationalism Agent: Detects emotional and exaggerated rhetoric
4. Malicious Account Agent: Identifies linguistic markers of inauthentic behavior
5. Naive Realism Agent: Measures absolutist language and opinion-as-fact representation

To help coordinate these agents and synthesize results, the system also utilizes an orchestrator and merger agent.
- Orchestrator: A SequentialAgent that calls upon the specialized sub-agents to perform the analysis and score the given article
- Merger Agent: Combines the analysis results from the 4 factuality factor agents and gives the article a final truthfulness label

### Prompt Experimentation
Each factuality factor agent includes seven prompt variants for systematic testing:
- Base, Chain-of-Thought, Fractal COT (3 variants), Function Calling, In-Context Learning

### Current Status
Testing Phase: All articles are being tested by selecting the orchestrator via the ADK web interface. Single agents can also be used from the dropdown menu and tested by pasting articles into the chat. Additionally, we are currently working on hill climbing for Fractal COT.

Next Steps: Connect the agent piepeline to the Flask UI for end-to-end verification workflow.


## Repository Structure
```text
DSC180A-Q1Project/
├── data/
│   ├── article.txt                   # Article text for ingestion
│   ├── ground_truth.csv              # Hand-labeled articles with appropriate factuality factor scores
│   ├── politifact.csv                # Manually scraped data from Politifact.org to augment LIAR-PLUS dataset
│   ├── train_set.csv                 # Training set with new scraped data
│   ├── train2.tsv                    # Original LiarPLUS train set
│   ├── val_set.csv                   # Validation set with new scraped data
│   ├── val2.tsv                      # Original LiarPLUS validation set
│   ├── test_set.csv                  # Test set with new scraped data
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
│   ├── metrics.ipynb                 # Accuracy scores for predictive models and LLM models
│   ├── model_accuracy.ipynb          # Various performance metrics for baseline predictive models
│   ├── prompt_results.ipynb          # Prompting experiment visualizations
│   ├── prompting.ipynb               # Contains 20 incremental prompts refining the model 
│   └── scraped_data.ipynb            # Additonal scraped data from Politifact added to LiarPLUS
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
│   ├── articles.py                   # Article ingestion, preprocessing
│   ├── predictive_models.py          # ML/LLM-based prediction pipeline
│   ├── script.py                     # Main script to run full pipeline
│   └── config.json                   # Config settings for models and pipeline
│
├── webapp/                           # Flask-based UI for demo interactions
│   ├── app.py                        # Flask entrypoint
|   ├── prompts/
│   │   ├── base.txt                      # Base prompt for LLM 
│   │   ├── chain_of_thought.txt          # LLM prompt incorporating chain of thought
│   │   └── factal_chain_of_thought.txt   # LLM prompt incorporating fractal chain of thought
|   ├── results/
│   │   ├── base_outputs.csv              # Outputs saved from running LLM with base prompt
│   │   ├── cot_outputs.csv               # Outputs saved from running LLM with chain of thought prompt
│   │   └── fcot_outputs.csv              # Outputs saved from running LLM with fractal chain of thought prompt
│   ├── static/
│   │   └── style.css                 # CSS styling for UI
│   └── templates/
│       └── index.html                # Main UI page
│
├── .gitignore
├── README.md
└── requirements.txt                  # Requirements for environment
```

## Dataset
This project uses the LIAR-PLUS dataset, an extended version of the original LIAR dataset. We augmented this dataset with more recent
scraped data from PolitiFact.
This includes labeled political statements along with metadata such as subjects, speakers, party affiliations, and justifications.

### Dataset Summary
| Split | File | Description |
|-------|------|--------------|
| **Train** | `train_set.csv` | Used to train all factuality models. |
| **Validation** | `val_set.csv` | Used for tuning and intermediate evaluation. |
| **Test** | `test_set.csv` | Used for final evaluation and analysis. |

## Installation

### Clone the repository
```bash
git clone https://github.com/JacquelynGarcia/DSC180A-Q1Project.git
cd Multi-Agent-Misinformation-Analysis-System
```

### Create virtual environment
```bash
python -m venv venv

source venv/bin/activate # Mac/Linux
venv\Scripts\activate # Windows
```

### Create a .env file
Create a `.env` file in the `webapp` folder and paste your API key from [Google AI Studio](https://aistudio.google.com).
```env
GEMINI_API_KEY=YOUR_API_KEY
```

### Install dependencies
```bash
pip install -r requirements.txt
```

You're ready to go!

## Requirements
To ensure this project is reproducible, this project uses the following Python libraries and versions:

```lua
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

If you encounter an OpenMP error on macOS when running XGBoost, ensure homebrew is installed and install the OpenMP runtime:
```bash
brew install libomp
```

## Running the Pipeline
Once the dataset and environment are set up, you can execute all four factuality models in sequence using the main script.

### Run the project
From the project root directory, run:

```bash
python -m src.script
```

If properly installed, the example console output should contain the following:
```
Datasets loaded successfully.
Frequency Heuristic model trained.
Sensationalism model trained.
Malicious Account model trained.
Naive Realism model trained.

Analyzing article...

Article Analysis Results:
Frequency Heuristic Level: 0
Frequency Heuristic Score: 0.305
Sensationalism Level: 2
Sensationalism Score: 0.835
Malicious Account Level: 0
Malicious Account Score: 0.225
Naive Realism Level: 1
Naive Realism Score: 0.566
```

## Agent-Based Verification
The agent system is currently in testing mode using the ADK web interface.

### Testing Workflow
1. Run the ADK web interface

```bash
adk web ff_agents
```
2. Select the orchestrator agent from the top left drop down menu
3. Paste an article into the chat to analyze

### Changing Prompt Variants
To experiment with different prompts for the factuality agents, edit the `agent.py` file and modify the prompt loading line:
```bash
prompt = load_prompt(_AGENT_DIR, "file_name.txt")
```
Available prompt files: `base.txt`, `cot.txt`, `fcot.txt`, `fcot2.txt`, `fcot3.txt`, `function_calling.txt`, `icl.txt`

Note: Full agent integration with the Flask web interface is in development. Each factuality agent supports multiple prompt variants for experimentation.

## Running the Web Application
The Flask UI lets you do the following:
- Paste an article link
- Paste article text
- View scores for each factuality factor
- View reasoning and confidence percentages
- Use a Clear button to analyze a new article
- Automatically saves every analysis to `results/data_outputs.csv`

### 1. Navigate to the webapp directory
```bash
cd webapp
```

### 2. Start the Flask server
```bash
python app.py
```

### 3. Opent the UI in your browser
```cpp
http://127.0.0.1:5000
```
Your interactive Narrative Integrity Analyzer is now running!

## Model Summaries
Each model focuses on a different factuality factor within political statements, capturing linguistic, contextual, or behavioral patterns associated with truthfulness and bias.

### Frequency Heuristic Model
Goal: Detect linguistic cues that may indicate exaggeration or misinformation through overuse of buzzwords and repetition.  

Features:
- TF-IDF mean  
- Average word frequency  
- Buzzword count  
- Repetition ratio  

Model: `RandomForestClassifier`  
Outputs:
- `predicted_label`
- `frequency_heuristic_score` - probability of label confidence  


### Malicious Account Model
Goal: Goal: Identify linguistic and behavioral traces aligned with inauthentic or “malicious account” behavior

Features:
- TF-IDF mean  
- Average token length
- Repitition score
- Link count
- Hashtag + mention count
- Punctuation ratio
- Uppercase ratio

Model: `RandomForestClassifier` within a `scikit-learn` Pipeline using `StandardScaler` 

Outputs:
- `predicted_malicious_account`
- `malicious_account_score` - probability of label confidence

### Sensationalism Model
Goal: Identify emotional, exaggerated, or dramatic tones that make a statement "sensational".

Features:
- Exclamation count (`!`)  
- Number of ALLCAPS words  
- Sensational keywords
- Sentiment polarity and subjectivity 
- Metadata such as `speaker`, `party`, and `context`  

Model: `XGBoost` within a `scikit-learn` Pipeline using `ColumnTransformer`  
Outputs:
- `predicted_sensationalism`
- `sensationalism_score` - probability of label confidence

### Naive Realism Model
Goal: Measure how strongly a statement presents opinion as fact through absolutist phrasing, lack of hedging, and dismissive language.

Features:
- Absolute-language ratio
- Cautious-language ratio
- Dismissive term count

Model: `XGBoost` Pipeline  
Outputs:
- `predicted_naive_realism`
- `naive_realism_score` - probability of label confidence

## Agent System Details

### Claim Extraction Agent
Powered by gemini-3-flash, this agent coordinates claim extraction and verification.

#### Extraction Capabilities
- Extracts 3-7 atomic, factual claims per article
- Filters out opinions, rhetoric, and speculation
- Produces self-contained, testable statements
- Uses a single instruction

### Verification via Tool
- Uses the Claim Verification Agent as a tool to validate extracted claims
- The Claim Verification Agent leverages Google Search to find authoritative sources
- Evaluates evidence from multiple perspectives
- Assigns verification status and confidence levels
- Returns detailed verification reports with source URLs

#### Verification Statuses
SUPPORTED: Multiple credible sources confirm

REFUTED: Credible sources contradict

PARTIALLY_SUPPORTED: Mixed evidence

UNVERIFIABLE: Insufficient information

CONFLICTING: Sources disagree

**Architecture Note**: This approach replaces traditional RAG with direct Google Search integration for more current fact-checking.

### Factuality Factor Agents
Each of the four factuality factors has been reimplemented as an agent with experimental prompt variants.

**Frequency Heuristic Agent**

Analyzes repetition patterns, buzzwords, and TF-IDF signals that may indicate manipulation or "truthiness."

**Sensationalism Agent**

Detects emotional language, exaggeration, ALLCAPS usage, and dramatic rhetoric.

**Malicious Account Agent**

Identifies linguistic markers associated with bot-like behavior, spam patterns, and inauthentic messaging.

**Naive Realism Agent**

Measures absolutist phrasing, lack of hedging, and dismissive language that presents opinion as fact.

## Note on Exploratory Models and Methodology Evolution

During the early stages of this project, our team explored a broader set of factuality factors through exploratory data analysis and experimental modeling found within the `eda_visualization.ipynb` notebook. This notebook includes preliminary implementations of **Credibility** and **Echo Chamber** models, along with earlier versions of the **Frequency Heuristic** and **Sensationalism** models.

As our methodology matured, we refined our framework to focus on four primary factuality dimensions for the final production pipeline which included **Frequency Heuristic**, **Sensationalism**, and the newly added **Naive Realism**, and **Malicious Account**.

While **Credibility** and **Echo Chamber** are not included in the finalized model pipeline, exploratory versions of these models remain available in the `notebooks/` directory. Users who wish to extend the project, compare modeling strategies, or incorporate additional factuality factors are welcome to experiment with these earlier models.

## License
This project is part of the DSC180A Capstone course at UC San Diego.
