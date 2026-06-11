#!/usr/bin/env python3
"""
Offline script to build the FAISS semantic search index.

Run once after deployment, or after approved cases are merged:
    python scripts/build_embeddings.py

Uses the same CSV path as the main application.
"""

import sys
import os

# Add project root to path so we can import services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from services.semantic_search import SemanticSearchService


def main():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            '10yearsdata_cleaned.csv')

    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        sys.exit(1)

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")

    embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'embeddings')

    service = SemanticSearchService(embeddings_dir=embeddings_dir)
    service.build_embeddings(df)

    print("\nDone! FAISS index saved to:")
    print(f"  {service.index_path}")
    print(f"  {service.meta_path}")


if __name__ == '__main__':
    main()
