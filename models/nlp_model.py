# models/nlp_model.py

# models/nlp_model.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"NLP running on: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()


# -------------------------------------------------
# SINGLE TEXT SENTIMENT (for compatibility)
# -------------------------------------------------
def get_sentiment_score(text):
    """
    Returns numeric weighted sentiment score for single text.
    """

    if not isinstance(text, str):
        text = str(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)

    prediction = prediction.item()
    confidence = confidence.item()

    if prediction == 2:      # positive
        return confidence
    elif prediction == 0:    # negative
        return -confidence
    else:
        return 0.0


# -------------------------------------------------
# BATCH SENTIMENT (recommended)
# -------------------------------------------------
def get_batch_sentiment(text_list, batch_size=16):
    """
    Returns weighted sentiment scores:
    positive -> +confidence
    negative -> -confidence
    neutral  -> 0
    """

    scores = []

    for i in range(0, len(text_list), batch_size):

        batch = text_list[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)
        confidence, predictions = torch.max(probs, dim=1)

        predictions = predictions.cpu().numpy()
        confidence = confidence.cpu().numpy()

        for pred, conf in zip(predictions, confidence):
            if pred == 2:
                scores.append(conf)
            elif pred == 0:
                scores.append(-conf)
            else:
                scores.append(0.0)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return scores