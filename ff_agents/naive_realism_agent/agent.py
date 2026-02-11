from pathlib import Path

from google.adk.agents import LlmAgent

try:
    from tools.model_predictions import get_naive_realism_prediction
    from tools.prompts import load_prompt
except ImportError:
    from ff_agents.tools.model_predictions import get_naive_realism_prediction
    from ff_agents.tools.prompts import load_prompt

_AGENT_DIR = Path(__file__).resolve().parent

prompt = load_prompt(_AGENT_DIR, "base.txt")

naive_realism_agent = LlmAgent(
    model='gemini-3-flash-preview',
    name='naive_realism_agent',
    description='Analyzes an article for naive realism and gives it a naive realism score ' \
    'from 0-2 based on the analysis.',
    instruction=prompt,
    output_key='naive_realism_analysis',
    tools=[get_naive_realism_prediction],
)

