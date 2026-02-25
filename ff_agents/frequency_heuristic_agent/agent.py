from pathlib import Path
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

try:
    from tools.model_predictions import get_frequency_heuristic_prediction
    from tools.prompts import load_prompt
except ImportError:
    from ff_agents.tools.model_predictions import get_frequency_heuristic_prediction
    from ff_agents.tools.prompts import load_prompt

_AGENT_DIR = Path(__file__).resolve().parent

prompt = load_prompt(_AGENT_DIR, "fcot4.txt")

freq_heuristic_agent = LlmAgent(
    model='gemini-3-flash-preview',
    name='freq_heuristic_agent',
    description='Analyzes an article for repetition, a traceable origin, and evidence verification, '
    'and gives it a frequency heuristic score from 0-2 based on analysis.',
    instruction=prompt,
    output_key='frequency_heuristic_analysis',
    tools=[get_frequency_heuristic_prediction],
)

root_agent = freq_heuristic_agent
a2a_app = to_a2a(root_agent, port=8001)

