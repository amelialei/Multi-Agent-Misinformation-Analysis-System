from google.adk.agents import LlmAgent

parallel_analysis_agent = ParallelAgent(
    model='gemini-2.5-flash',
    name='parallel_analysis_agent',
    sub_
)
root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)
