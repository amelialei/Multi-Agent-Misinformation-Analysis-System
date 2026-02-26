from google.adk.agents import ParallelAgent
from google.adk.agents import SequentialAgent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from frequency_heuristic_agent.agent import freq_heuristic_agent
from malicious_account_agent.agent import malicious_acc_agent
from naive_realism_agent.agent import naive_realism_agent
from sensationalism_agent.agent import sensationalism_agent
from merger_agent.agent import merger_agent
from claim_extraction_agent.agent import claim_extraction_agent, claim_verification_agent

claim_pipeline = SequentialAgent(
    name='claim_pipeline',
    sub_agents=[claim_extraction_agent, claim_verification_agent],
    description='Extracts and verifies factual claims from the article.',
)

parallel_analysis_agent = ParallelAgent(
    name='parallel_analysis_agent',
    sub_agents=[
        freq_heuristic_agent, 
        malicious_acc_agent, 
        naive_realism_agent, 
        sensationalism_agent,
        ],
    description='Runs multiple factuality factor analysis agents in parallel to gather scores and insights on an article.',
)

root_agent = SequentialAgent(
    name='root_agent',
    sub_agents=[claim_pipeline, parallel_analysis_agent, merger_agent],
    description='Full factuality analysis pipeline.',
)
