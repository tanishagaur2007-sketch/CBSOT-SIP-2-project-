"""
Regenerates df.pkl exactly as used by the AI Search Intelligence System.

Usage:
    pip install datasets pandas
    python build_df_pkl.py

Produces df.pkl in the current directory.
"""

from datasets import load_dataset
import pandas as pd

print("Downloading dataset...")
dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")

print("Building DataFrame...")
df = pd.DataFrame(dataset)[["title", "abstract"]].head(15000)
df["paper_text"] = (
    (df["title"] + df["abstract"])
    .str.replace("\n", " ", regex=False)
    .str.strip()
)

df.to_pickle("df.pkl")
print(f"Saved df.pkl with shape {df.shape}")
