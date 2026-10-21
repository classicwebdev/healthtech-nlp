"""
Clinical NLP Inference Engine
-----------------------------
Encapsulates Hugging Face transformers model loading and entity extraction logic.
"""

from typing import List, Dict, Any
from transformers import pipeline

class ClinicalNEREngine:
    def __init__(self, model_name: str = "samrawal/bert-base-uncased_clinical-ner"):
        """
        Loads the model into RAM/VRAM once during API startup.
        """
        print(f"Loading clinical model weights from: {model_name}...")
        self.ner_pipeline = pipeline(
            task="ner",
            model=model_name,
            aggregation_strategy="simple"
        )

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Processes clinical text string and returns formatted entity dictionaries.
        """
        if not text.strip():
            return []
            
        raw_results = self.ner_pipeline(text)
        
        cleaned_entities = []
        for entity in raw_results:
            cleaned_entities.append({
                "entity": entity["word"],
                "category": entity["entity_group"],
                "score": round(float(entity["score"]), 4)
            })
            
        return cleaned_entities