from pathlib import Path
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

try:
    from tools.model_predictions import get_malicious_account_prediction
    from tools.prompts import load_prompt
except ImportError:
    from ff_agents.tools.model_predictions import get_malicious_account_prediction
    from ff_agents.tools.prompts import load_prompt

_AGENT_DIR = Path(__file__).resolve().parent

prompt = load_prompt(_AGENT_DIR, "fcot4.txt")

malicious_acc_agent = LlmAgent(
    model='gemini-3-flash-preview',
    name='malicious_acc_agent',
    description='Analyzes an article for malicious content and gives it a malicious account score ' \
    'from 0-2 based on the analysis.',
    instruction=prompt,
    output_key='malicious_account_analysis',
    tools=[get_malicious_account_prediction],
)

root_agent = malicious_acc_agent
a2a_app = to_a2a(root_agent, port=8002)