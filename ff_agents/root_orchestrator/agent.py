from google.adk.agents import ParallelAgent
from google.adk.agents import SequentialAgent
from ff_agents.frequency_heuristic_agent.agent import freq_heuristic_agent
from ff_agents.malicious_account_agent.agent import malicious_acc_agent
from ff_agents.naive_realism_agent.agent import naive_realism_agent
from ff_agents.sensationalism_agent.agent import sensationalism_agent

parallel_analysis_agent = ParallelAgent(
    model='gemini-2.5-flash',
    name='parallel_analysis_agent',
    subagents=[freq_heuristic_agent, malicious_acc_agent, naive_realism_agent, sensationalism_agent],
    description='Runs multiple factuality factor analysis agents in parallel to gather scores and insights on an article.',
)
# root_agent = SequentialAgent(
#     model='gemini-2.5-flash',
#     name='root_agent',
#     description='A helpful assistant for user questions.',
#     instruction='Answer user questions to the best of your knowledge',
# )
