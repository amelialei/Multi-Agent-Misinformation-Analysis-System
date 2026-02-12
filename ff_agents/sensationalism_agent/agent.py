from pathlib import Path

from google.adk.agents import LlmAgent

try:
    from tools.model_predictions import get_sensationalism_prediction
    from tools.prompts import load_prompt
except ImportError:
    from ff_agents.tools.model_predictions import get_sensationalism_prediction
    from ff_agents.tools.prompts import load_prompt

_AGENT_DIR = Path(__file__).resolve().parent

prompt = load_prompt(_AGENT_DIR, "fcot2.txt")

sensationalism_agent = LlmAgent(
    model='gemini-3-flash-preview',
    name='sensationalism_agent',
    description='Analyzes an article for sensationalism and gives it a sensationalism score ' \
    'from 0-2 based on the analysis.',
    instruction=prompt,
    output_key='sensationalism_analysis',
    tools=[get_sensationalism_prediction],
)

