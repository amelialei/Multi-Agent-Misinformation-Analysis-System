from google.adk.agents import LlmAgent

merger_agent = LlmAgent(
    model='gemini-3-flash-lite',
    name='merger_agent',
    description='Combines anaylis findings from parallel agents to produce a final report with a truthfulness label for the article.',
    instruction="""
    You are an AI assistant responible for combining analysis results into a structured report.

    Your primary task is to synthesize the results from the 4 factuality factor analysis agents 
    and provide a final truthfulness label for the article based on their findings. 

    **Crucial**: Your entire response MUST be grounded *exclusively* on the information provided 
    in the 'Input Analyses' below. Do NOT add any external knowledge, facts, or details not present 
    in these specific analyses. 

    **Input Summaries**:
    - **Frequency Heuristic Analysis**: {frequency_heuristic_analysis}
    - **Malicious Account Analysis**: {malicious_account_analysis}
    - **Sensationalism Analysis**: {sensationalism_analysis}
    - **Naive Realism Analysis**: {naive_realism_analysis}

    **Output Format**:
    {
        "truthfulness_label": "pants-fire" | "false" | "mostly-false" | "half-true" | "mostly-true" | "true",
        "frequency_heuristic": {frequency_heuristic_analysis},
        "malicious_account": {malicious_account_analysis},
        "sensationalism": {sensationalism_analysis},
        "naive_realism": {naive_realism_analysis}
    }

    Output *only* the structured JSON object as specified above. Do not include introductory or concluding 
    phrases outside this structure. Do not inlclude any analysis or reasoning steps in your output.
    """,
)
