from google.adk.agents import LlmAgent

merger_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='merger_agent',
    description='A helpful assistant for user questions.',
    # instruction="""
    # You are an AI assistant responible for combining analysis results into a structured report.

    # Your primary task is to 


    # """,
)
