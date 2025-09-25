import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_CSV = "data/founders.csv"
MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs("data", exist_ok=True)

def load_data():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} not found. Run generate_dataset.py first.")
    df = pd.read_csv(DATA_CSV)
    for c in ["idea","about","keywords","founder_name","company","location","role","id"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    return df

def main():
    print("Loading data...")
    df = load_data()
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    fields = ["idea","about","keywords","founder_name","company","location","role"]
    for f in fields:
        texts = df[f].tolist()
        emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        np.save(f"data/emb_{f}.npy", emb)

    combined_texts = (
        df["idea"].astype(str) + ". " +
        df["about"].astype(str) + ". " +
        df["keywords"].astype(str)
    ).tolist()
    emb_comb = model.encode(combined_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    np.save("data/emb_combined.npy", emb_comb)

    df.to_csv("data/meta_founders.csv", index=False)
    print("Indexing complete!")

if __name__ == "__main__":
    main()
