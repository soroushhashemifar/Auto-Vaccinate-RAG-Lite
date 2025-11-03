from evaluation import evaluate_rag
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("out/bandit_eval_dataset.csv", sep="\t")

    original_response_indices = []
    for idx, chunk_df in df.groupby("index"):
        original_response_indices.append(chunk_df.index[0])

    df = df.iloc[original_response_indices]
    evaluate_rag(df)