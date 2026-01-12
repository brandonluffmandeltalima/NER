"""
Unified FastAPI Application
- Email Relevancy Classifier
- SpaCy NER Model
"""

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import spacy
import uvicorn

from email import policy
from email.parser import BytesParser

# Local imports
from model import EmailClassifier
from preprocessor import TextPreprocessor
from email_ingestion import OutlookGraphDownloader
import nltk
nltk.download("punkt")
nltk.download('punkt_tab')

# ----------------------------
# Initialize FastAPI
# ----------------------------
app = FastAPI(
    title="Unified NLP API",
    description="API that includes Email Relevancy Classification + Named Entity Recognition",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Global variables / Models
# ----------------------------
classifier = None
preprocessor = None
nlp = None

# Lazy-load Email Classifier
def get_classifier():
    global classifier, preprocessor
    if classifier is None:
        print("Loading Email Classifier Model...")
        classifier = EmailClassifier()
        classifier.load_model("models/email_classifier.pkl")
        preprocessor = TextPreprocessor(lowercase=True, remove_numbers=False)
        print("Email Classifier loaded.")
    return classifier, preprocessor

# Lazy-load spaCy NER
def get_nlp():
    global nlp
    if nlp is None:
        import spacy
        MODEL_PATH = os.path.join("output", "model-best")
        print("Loading spaCy NER model...")
        nlp = spacy.load(MODEL_PATH)
        print("spaCy NER loaded.")
    return nlp



# ----------------------------
# Pydantic Models (Email Classifier)
# ----------------------------
class EmailInput(BaseModel):
    subject: str
    body: str
    email_id: str | None = None


class EmailBatchInput(BaseModel):
    emails: List[EmailInput]


class PredictionOutput(BaseModel):
    email_id: str | None
    label: str
    is_relevant: bool
    confidence: float
    probabilities: dict


class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ----------------------------
# Pydantic Models (NER)
# ----------------------------
class TextInput(BaseModel):
    text: str


# ----------------------------
# Startup – Load Both Models
# ----------------------------
# @app.on_event("startup")
# async def load_models():
#     global classifier, preprocessor, nlp

#     print("Loading Email Classifier Model...")
#     classifier = EmailClassifier()

#     try:
#         classifier.load_model("models/email_classifier.pkl")
#         print("Email classifier loaded.")
#     except FileNotFoundError:
#         print("ERROR: Missing model at models/email_classifier.pkl")
#         raise

#     preprocessor = TextPreprocessor(lowercase=True, remove_numbers=False)

#     print("Loading spaCy NER model...")
#     MODEL_PATH = os.path.join("output", "model-best")
#     nlp = spacy.load(MODEL_PATH)
#     print("spaCy NER loaded.")


# ----------------------------
# GENERAL ENDPOINTS
# ----------------------------
@app.get("/")
async def home():
    return {
        "message": "Unified NLP API",
        "email_classifier": "/classify",
        "email_classifier_batch": "/classify/batch",
        "ner": "/ner",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "healthy",
        "model_loaded": classifier is not None and classifier.is_trained
    }


# ----------------------------
# EMAIL CLASSIFIER ENDPOINTS
# ----------------------------
# @app.post("/classify", response_model=PredictionOutput)
# async def classify_email(email: EmailInput):

#     if not classifier or not classifier.is_trained:
#         raise HTTPException(status_code=503, detail="Model not loaded")

#     try:
#         text = preprocessor.preprocess_email(email.subject, email.body, remove_stops=False)
#         result = classifier.predict([text])[0]
#         result["email_id"] = email.email_id
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/classify", response_model=PredictionOutput)
async def classify_email(email: EmailInput):
    classifier_obj, preprocessor_obj = get_classifier()  # lazy load

    if not classifier_obj.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = preprocessor_obj.preprocess_email(email.subject, email.body, remove_stops=False)
    result = classifier_obj.predict([text])[0]
    result["email_id"] = email.email_id
    return result


# @app.post("/classify/batch", response_model=BatchPredictionOutput)
# async def classify_email_batch(batch: EmailBatchInput):

#     if not classifier or not classifier.is_trained:
#         raise HTTPException(status_code=503, detail="Model not loaded")

#     if not batch.emails:
#         raise HTTPException(status_code=400, detail="No emails provided")

#     try:
#         texts = []
#         ids = []

#         for email in batch.emails:
#             texts.append(
#                 preprocessor.preprocess_email(email.subject, email.body, remove_stops=False)
#             )
#             ids.append(email.email_id)

#         results = classifier.predict(texts)

#         for r, eid in zip(results, ids):
#             r["email_id"] = eid

#         return {
#             "predictions": results,
#             "total": len(results)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Batch classification error: {str(e)}")

@app.post("/classify/batch", response_model=BatchPredictionOutput)
async def classify_email_batch(batch: EmailBatchInput):
    classifier_obj, preprocessor_obj = get_classifier()  # lazy load

    if not classifier_obj.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not batch.emails:
        raise HTTPException(status_code=400, detail="No emails provided")

    texts = [preprocessor_obj.preprocess_email(e.subject, e.body, remove_stops=False) for e in batch.emails]
    ids = [e.email_id for e in batch.emails]
    results = classifier_obj.predict(texts)
    for r, eid in zip(results, ids):
        r["email_id"] = eid

    return {"predictions": results, "total": len(results)}


# ----------------------------
# NER ENDPOINT
# ----------------------------
# @app.post("/ner")
# async def ner_extract(input: TextInput):

#     try:
#         doc = nlp(input.text)
#         return {
#             "entities": [
#                 {
#                     "text": ent.text,
#                     "label": ent.label_,
#                     "start": ent.start_char,
#                     "end": ent.end_char
#                 }
#                 for ent in doc.ents
#             ]
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"NER error: {str(e)}")


@app.post("/ner")
async def ner_extract(input: TextInput):
    nlp_obj = get_nlp()  # lazy load

    doc = nlp_obj(input.text)
    return {
        "entities": [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
    }


# ----------------------------
# Email Ingestion
# ----------------------------

def get_plain_text_from_mime(mime_bytes):
    """Extract plain text body from MIME content"""
    try:
        msg = BytesParser(policy=policy.default).parsebytes(mime_bytes)
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            return msg.get_payload(decode=True).decode('utf-8', errors='ignore')
    except:
        return ""

@app.get("/emails")
async def get_emails():
    try:
        downloader = OutlookGraphDownloader()
        if not downloader.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")

        messages = downloader.get_messages()
        if not messages:
            return JSONResponse(content={"emails": []})

        first_five = messages[1:6]
        emails_data = []

        for msg in first_five:
            msg_id = msg.get("id")
            mime_content = downloader.get_message_mime(msg_id)
            body = get_plain_text_from_mime(mime_content) if mime_content else ""

            emails_data.append({
                "id": msg_id,
                "subject": msg.get("subject", ""),
                "from": msg.get("from", {}).get("emailAddress", {}).get("address", ""),
                "to": [r['emailAddress']['address'] for r in msg.get("toRecipients", [])],
                "receivedDateTime": msg.get("receivedDateTime", ""),
                "body": body
            })

        return JSONResponse(content={"emails": emails_data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ----------------------------
# Run Server - Local Dev
# ----------------------------
if __name__ == "__main__":
    print("Starting Unified NLP API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

