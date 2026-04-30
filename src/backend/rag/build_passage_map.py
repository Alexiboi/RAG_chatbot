# build_passage_map.py — run this ONCE when you build your index
import asyncio
import json
from src.backend.rag.index_utils import TRANSCRIPT_SEARCH_CLIENT

def build_passage_map(output_path: str = "./evaluation/passage_map.json"):
    passage_map = {}
    duplicate_ids = []
    skip = 0
    batch_size = 100

    while True:
        results = list(TRANSCRIPT_SEARCH_CLIENT.search(
            search_text="*",
            select=["id", "content"],
            top=batch_size,
            skip=skip,
            include_total_count=True
        ))
        
        if not results:
            break

        for r in results:
            chunk_id = r["id"]
            normalised = " ".join(r["content"].split()).lower()
            
            if chunk_id in passage_map:
                duplicate_ids.append(chunk_id)
            
            passage_map[chunk_id] = normalised

        skip += len(results)
        print(f"Fetched {skip} chunks so far...")

        if len(results) < batch_size:
            break

    print(f"\nTotal duplicated IDs: {len(duplicate_ids)}")
    print(f"Example duplicate IDs: {duplicate_ids[:5]}")

    with open(output_path, "w") as f:
        json.dump(passage_map, f)

    print(f"Saved {len(passage_map)} chunks to {output_path}")
if __name__ == "__main__":
    build_passage_map()
