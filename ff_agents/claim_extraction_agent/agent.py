from google.adk.agents import LlmAgent

claim_extraction_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='claim_extraction_agent',
    description='Extracts verifiable factual claims from the article for downstream retrieval and analysis.',
    instruction="""
    You are an expert at identifying factual claims suitable for external verification.

    Your task is to extract concise, atomic factual claims from the provided article.

    ## Claim Extraction Rules
    - Extract between 3 and 7 claims.
    - Each claim must be:
        - Factual (not opinion or rhetoric)
        - Testable against external sources
        - Self-contained and unambiguous
    - Do NOT include:
        - Emotional language
        - Predictions
        - Value judgments
        - Questions

    ## Examples
    Article: The CDC confirmed that vaccine uptake increased by 12% in 2023.
    Claim: "The CDC reported a 12% increase in vaccine uptake in 2023."

    Article: Experts warn this policy could destroy the economy.
    Do not extract — speculative.

    ## Output Format:
    {
        "claims": [
            "Claim 1",
            "Claim 2",
            "Claim 3"
        ]
    }

    RETURN ONLY VALID JSON. DO NOT USE MARKDOWN. DO NOT USE ```json OR ANY CODE FENCES.
    """,
    output_key="extracted_claims"
)
