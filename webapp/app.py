from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from google.genai import types
import json
import pandas as pd
import uuid
import sys
import asyncio

# Add project root to Python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

# ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# Importing orchestrator root agent
sys.path.append(os.path.join(ROOT, "ff_agents"))
from orchestrator.agent import root_agent

load_dotenv()
app = Flask(__name__)

# ADK Runner Setup
APP_NAME = "factuality_app"
session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

# Agent Invocation
def run_agent_pipeline(article_text: str) -> dict:
    """
    Runs the full agent pipeline via ADK Runner and returns the parsed JSON result.
    """
    async def _run():
        session_id = str(uuid.uuid4())
        user_id = "flask_user"

        # Await the async session creation
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

        message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=article_text)],
        )

        final_response = None

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            if event.is_final_response():
                final_response = event.content.parts[0].text

        return final_response

    final_response = asyncio.run(_run())

    if not final_response:
        raise ValueError("Agent pipeline returned no response.")

    clean = final_response.strip()
    start = clean.find("{")
    end = clean.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in agent output: {clean}")

    return json.loads(clean[start:end+1])

# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")

def save_to_csv(article_url, parsed):
    run_id = str(uuid.uuid4())
    row = {
        "id": run_id,
        "url": article_url,
        "freq_score":  parsed.get("frequency_heuristic", {}).get("score"),
        "mal_score":   parsed.get("malicious_account", {}).get("score"),
        "sens_score":  parsed.get("sensationalism", {}).get("score"),
        "naive_score": parsed.get("naive_realism", {}).get("score"),
        "truthfulness_label": parsed.get("truthfulness_label"),
    }
    df_row = pd.DataFrame([row])
    csv_path = "results/fcot_outputs_pt2.csv"
    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)

@app.route("/score", methods=["POST"])
def score():
    article_text = request.form.get("article", "")
    article_url  = request.form.get("article_url", "")

    if not article_text.strip():
        return jsonify({"error": "No article text provided"}), 400

    try:
        parsed = run_agent_pipeline(article_text)
        save_to_csv(article_url, parsed)
        return jsonify(parsed)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON from agent: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,
    )
