import json
from src.predictive_models import (
    load_datasets, 
    build_frequency_model, 
    build_sensationalism_model,
    build_malicious_account_model,
    build_naive_realism_model
)

from src.articles import (
    evaluate_article
)

def main():
    with open("src/config.json", "r") as f:
        cfg = json.load(f)

    train_path = "data/train_set.csv"
    val_path = "data/val_set.csv"
    test_path = "data/test_set.csv"
    df_train, df_val, df_test = load_datasets(
        train_path, val_path, test_path
    )
    print("Datasets loaded successfully.")

    freq_cfg = cfg["models"]["frequency"]
    freq_model = build_frequency_model(df_train, **freq_cfg)
    print("Frequency Heuristic model trained.")

    sens_cfg = cfg["models"]["sensationalism"]
    sens_model = build_sensationalism_model(
        df_train,**sens_cfg)
    print("Sensationalism model trained.")

    ma_cfg = cfg["models"]["malicious_account"]
    ma_model = build_malicious_account_model(
        df_train, **ma_cfg)
    print("Malicious Account model trained.")       

    nr_cfg = cfg["models"]["naive_realism"]
    nr_model = build_naive_realism_model(
        df_train, **nr_cfg)
    print("Naive Realism model trained.")

    # Use trained models to predict scores for new articles
    with open("data/article.txt", "r") as f:
        article_text = f.read()

    print("\nAnalyzing article...")

    article_analysis = evaluate_article(
        article_text,
        freq_model, 
        sens_model, 
        ma_model,
        nr_model
    )
    
    print(f"\nArticle Analysis Results:")
    print(f"Frequency Heuristic Level: {article_analysis['predicted_frequency_heuristic']}")
    print(f"Frequency Heuristic Score: {article_analysis['frequency_heuristic_score']:.3f}")
    print(f"Sensationalism Level: {article_analysis['predicted_sensationalism']}")
    print(f"Sensationalism Score: {article_analysis['sensationalism_score']:.3f}")
    print(f"Malicious Account Level: {article_analysis['predicted_malicious_account']}")
    print(f"Malicious Account Score: {article_analysis['malicious_account_score']:.3f}")
    print(f"Naive Realism Level: {article_analysis['predicted_naive_realism']}")
    print(f"Naive Realism Score: {article_analysis['naive_realism_score']:.3f}")


if __name__ == "__main__":
    main()