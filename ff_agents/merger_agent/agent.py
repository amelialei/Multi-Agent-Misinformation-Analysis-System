from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

merger_agent = LlmAgent(
    model='gemini-3-flash-preview',
    name='merger_agent',
    description='Combines analysis findings from parallel agents to produce a final JSON report.',
    instruction="""
    You are an AI assistant responsible for combining analysis results into a structured report.

    Synthesize the results from the 4 factuality factor analysis agents AND the claims 
    verification agent to produce a final truthfulness label and synopsis for the article.

    **Input Summaries**:
    - **Frequency Heuristic Analysis**: {frequency_heuristic_analysis}
    - **Malicious Account Analysis**: {malicious_account_analysis}
    - **Sensationalism Analysis**: {sensationalism_analysis}
    - **Naive Realism Analysis**: {naive_realism_analysis}
    - **Claims Verification**: {claims_verification}

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

    **Synopsis Writing Guide**:
    Your synopsis should be 4-5 sentences that:
    1. State the article's overall factuality in plain language
    2. Highlight 1-2 key concerns from the heuristic/stylistic factors (frequency, malicious accounts,
       sensationalism, naive realism)
    3. Reference specific claim verification outcomes — e.g. which claims were SUPPORTED, REFUTED, 
       or UNVERIFIABLE — and how they influenced the final label
    4. Conclude with what primarily drove the truthfulness label

    **How Claims Verification Should Influence the Label**:
    - If the majority of claims are REFUTED with HIGH confidence → lean toward false/pants-fire
    - If claims are mostly SUPPORTED → those findings can offset negative stylistic signals
    - If claims are UNVERIFIABLE or CONFLICTING → factor that uncertainty into a lower confidence score
    - PARTIALLY_SUPPORTED claims with other red flags → lean toward mostly-false or half-true

    **Output Format** (valid JSON only, no markdown, no code fences):
    {
        "truthfulness_label": "pants-fire|false|mostly-false|half-true|mostly-true|true",
        "synopsis": "4-5 sentence summary integrating heuristic signals and verified claim outcomes.",
        "frequency_heuristic": {frequency_heuristic_analysis},
        "malicious_account": {malicious_account_analysis},
        "sensationalism": {sensationalism_analysis},
        "naive_realism": {naive_realism_analysis}
    }
    """,
)

root_agent = merger_agent
a2a_app = to_a2a(root_agent, port=8006)

