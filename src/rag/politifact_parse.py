import pandas as pd

def load_politifact_corpus(csv_path="data/politifact.csv"):
    df = pd.read_csv(csv_path)

    corpus = []
    for _, row in df.iterrows():
        text_parts = []

        if "statement" in row and pd.notna(row["statement"]):
            text_parts.append(f"Claim: {row['statement']}")

        if "ruling" in row and pd.notna(row["ruling"]):
            text_parts.append(f"Verdict: {row['ruling']}")

        if "explanation" in row and pd.notna(row["explanation"]):
            text_parts.append(f"Explanation: {row['explanation']}")

        text = "\n".join(text_parts).strip()

        if not text:
            continue

        corpus.append({
            "text": text,
            "source": "politifact",
            "label": row.get("ruling", None),
            "url": row.get("url", None)
        })

    return corpus