import os
import yaml
from rapidfuzz import fuzz

class KB_handler:    
    kb = [] 

    @classmethod
    def _load_kb(cls):
        """Load Dawar KB from kb.yaml into memory."""
        kb_path = "prompts/KB.yaml"
        if not os.path.exists(kb_path):
            print("⚠️ No KB file found")
            cls.kb = []
            return cls.kb

        with open(kb_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        kb_entries = []
        for section, items in data.items():
            if isinstance(items, list):
                for item in items:
                    kb_entries.append({"section": section, "content": str(item)})
            else:
                kb_entries.append({"section": section, "content": str(items)})

        cls.kb = kb_entries
        return cls.kb

    @classmethod
    def _search_kb(cls, query, max_results: int = 3, threshold: int = 60, section: str = None):
        if not cls.kb:
            return None

        scored = []
        for entry in cls.kb:
            if section and entry["section"] != section:
                continue

            score = max(
                fuzz.partial_ratio(query, entry["content"]),
                fuzz.token_sort_ratio(query, entry["content"]),
                fuzz.token_set_ratio(query, entry["content"])
            )
            if score >= threshold:
                scored.append({"entry": entry, "score": score})

        # Sort results
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:max_results] if scored else None
