from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

merger_agent = LlmAgent(
    model='gemini-3-flash-preview',
    name='merger_agent',
    description='Combines analysis findings from parallel agents to produce a final JSON report.',
    instruction="""
    You are an AI assistant responsible for combining analysis results into a structured report.

    Synthesize the results from the 4 factuality factor analysis agents to produce a final 
    truthfulness label and synopsis for the article.

    **Input Summaries**:
    - **Frequency Heuristic Analysis**: {frequency_heuristic_analysis}
    - **Malicious Account Analysis**: {malicious_account_analysis}
    - **Sensationalism Analysis**: {sensationalism_analysis}
    - **Naive Realism Analysis**: {naive_realism_analysis}

    **Truthfulness Label Guide**:
    - pants-fire: Demonstrably false, often ridiculous
    - false: Not accurate
    - mostly-false: Contains a kernel of truth but is mostly false
    - half-true: Partially accurate but missing important context
    - mostly-true: Accurate but missing minor details
    - true: Accurate and complete

    **Confidence Score Guide** (0.0 to 1.0):
    - Based on clarity of evidence, consistency across factors, and strength of signals
    - HIGH confidence (0.8-1.0): Strong, consistent signals across all factors
    - MEDIUM confidence (0.5-0.79): Mixed or moderate signals
    - LOW confidence (0.2-0.49): Weak or conflicting signals

    **Output Format** (valid JSON only, no markdown, no code fences):
    {
        "truthfulness_label": "pants-fire|false|mostly-false|half-true|mostly-true|true",
        "synopsis": "3-4 sentence summary of the article's overall factuality, key concerns found, and what drove the final label.",
        "frequency_heuristic": {
            "score": 0|1|2,
            "reasoning": "brief explanation grounded in article text",
            "confidence": 0-100
        },
        "malicious_account": {
            "score": 0|1|2,
            "reasoning": "brief explanation grounded in article text",
            "confidence": 0-100
        },
        "sensationalism": {
            "score": 0|1|2,
            "reasoning": "brief explanation grounded in article text",
            "confidence": 0-100
        },
        "naive_realism": {
            "score": 0|1|2,
            "reasoning": "brief explanation grounded in article text",
            "confidence": 0-100
        }
    }
    """,
)

root_agent = merger_agent
a2a_app = to_a2a(root_agent, port=8006)

