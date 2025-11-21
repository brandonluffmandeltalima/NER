import spacy
from fastapi import FastAPI
import os
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load model (Render runs from root of repo)
MODEL_PATH = os.path.join("output", "model-best")
nlp = spacy.load(MODEL_PATH)

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

class TextInput(BaseModel):
    text: str

@app.post("/ner")
def ner(input: TextInput):
    doc = nlp(input.text)
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
