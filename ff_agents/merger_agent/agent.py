from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

merger_agent = LlmAgent(
    model='gemini-3-flash-preview',
    name='merger_agent',
    description='Combines analysis findings from parallel agents to produce a final JSON report.',
    instruction="""
    You are an AI assistant responsible for combining analysis results into a structured report.

    Synthesize the results from the 4 factuality factor analysis agents AND the claim 
    verification results to produce a final truthfulness label for the article.

    **Input Summaries**:
    - **Frequency Heuristic Analysis**: {frequency_heuristic_analysis}
    - **Malicious Account Analysis**: {malicious_account_analysis}
    - **Sensationalism Analysis**: {sensationalism_analysis}
    - **Naive Realism Analysis**: {naive_realism_analysis}

    Use the claim verification findings (available in session context) as additional 
    signal when assigning the truthfulness label — verified/refuted claims should 
    strengthen or lower the label accordingly.

    **Truthfulness Label Guide**:
    - pants-fire: Demonstrably false, often ridiculous
    - false: Not accurate
    - mostly-false: Contains a kernel of truth but is mostly false
    - half-true: Partially accurate but leaves out important context
    - mostly-true: Accurate but missing minor details
    - true: Accurate and complete

    **Output Format** (valid JSON only, no markdown, no code fences):
    {
        "truthfulness_label": "pants-fire|false|mostly-false|half-true|mostly-true|true",
        "frequency_heuristic": {
            "score": 0|1|2,
            "reasoning": "brief explanation"
        },
        "malicious_account": {
            "score": 0|1|2,
            "reasoning": "brief explanation"
        },
        "sensationalism": {
            "score": 0|1|2,
            "reasoning": "brief explanation"
        },
        "naive_realism": {
            "score": 0|1|2,
            "reasoning": "brief explanation"
        }
    }
    """,
)

root_agent = merger_agent
a2a_app = to_a2a(root_agent, port=8006)

