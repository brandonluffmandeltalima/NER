import spacy
from fastapi import FastAPI
import os

# Load model (Render runs from root of repo)
MODEL_PATH = os.path.join("output", "model-best")
nlp = spacy.load(MODEL_PATH)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify: ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"status": "running"}

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