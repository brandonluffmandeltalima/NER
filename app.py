import spacy
from fastapi import FastAPI
import os

# Load model (Render runs from root of repo)
MODEL_PATH = os.path.join("output", "model-best")
nlp = spacy.load(MODEL_PATH)

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/ner")
def ner(text: str):
    doc = nlp(text)
    return {
        "entities": [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents
        ]
    }