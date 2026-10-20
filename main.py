
from transformers import pipeline

def run_clinical_ner():
    print("Loading Clinical BERT model (samrawal/bert-base-uncased_clinical-ner)...")
    
    clinical_ner = pipeline(
        task="ner",
        model="samrawal/bert-base-uncased_clinical-ner",
        aggregation_strategy="simple" 
    )

    
    sample_note = (
        "Patient presents with severe migraine and persistent nausea. "
        "Administered 50mg Sumatriptan once daily. History of hypertension."
    )

    print("\n--- Clinical Input Note ---")
    print(sample_note)

    
    results = clinical_ner(sample_note)

    
    print("\n--- Extracted Medical Entities ---")
    print(f"{'ENTITY':<22} | {'CATEGORY':<15} | {'CONFIDENCE SCORE':<10}")
    print("-" * 55)
    
    for entity in results:
        word = entity["word"]
        category = entity["entity_group"]
        score = f"{float(entity['score']):.4f}"
        print(f"{word:<22} | {category:<15} | {score:<10}")

if __name__ == "__main__":
    run_clinical_ner()