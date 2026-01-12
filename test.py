import spacy

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "./output/model-best"  # path to your trained spaCy NER model
TEST_EMAIL = """
To: Daniel Kim

From: Olivia Parker
Subject: Anderson v. Westwood Builders – Discovery Update and Hearing Preparation

Dear Daniel,

I reviewed the contract documents with client Sarah Anderson on February 5, 2024, regarding the ongoing litigation against Westwood Builders. Counsel for Westwood Builders, Thomas Grant, indicated that additional evidence supporting their defense will be submitted by March 1, 2024 pursuant to California Code of Civil Procedure Section 2031.

We plan to file a motion to compel additional documents with the San Francisco County Superior Court if the evidence is not produced by the deadline. The court has scheduled a case management conference before Hon. Lisa M. Hernandez on March 22, 2024. During that hearing, we expect the court to discuss alleged delays in project completion and breach of contract claims.

Our legal strategy references Parker v. Oceanic Constructions and relies on the terms outlined in the Construction Agreement dated September 10, 2022. To support our position, we will submit internal communications between Sarah Anderson and Thomas Grant dated January 20, 2024 obtained from Legal Review Services Inc.

We also received a response to our motion for inspection of documents filed on February 10, 2024. Counsel for Westwood Builders objected, citing overbreadth and relevance concerns. We plan to address these objections in our reply.

Please confirm your availability on March 5, 2024 so we can prepare for the upcoming filing and coordinate with Sarah Anderson.

Best regards,
Olivia Parker
"""

# ---------------------------
# LOAD MODEL
# ---------------------------
nlp = spacy.load(MODEL_PATH)
doc = nlp(TEST_EMAIL)

# ---------------------------
# PRINT DETECTED ENTITIES
# ---------------------------
print("Detected entities:\n")
for ent in doc.ents:
    print(f"Text: {ent.text:50}  Label: {ent.label_}")
