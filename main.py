from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
import requests
import json
import re
import os
import uvicorn

app = FastAPI(
    title="Plastic Classification API",
    description="API for classifying plastic types and providing recycling insights",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model locally (no Google Drive fallback)
MODEL_PATH = "plastic.h5"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

CLASS_NAMES = [
    "HDPE (High-Density Polyethylene)",
    "OTHERS",
    "PET (polyethylene terephthalate)",
    "PP (polypropylene)",
    "PVC (Polyvinyl chloride)"
]

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Plastic Classification API",
        "endpoints": {
            "ping": "/ping (GET)",
            "predict": "/predict (POST)",
            "insights": "/insights (POST)"
        },
        "documentation": "/docs"
    }

@app.get("/ping", tags=["Health Check"])
async def ping():
    return {"status": "healthy", "message": "API is running"}

def read_file_as_image(data) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            "class": predicted_class,
            "confidence": confidence,
            "prediction_html": f"""
                <h2>Plastic Type Prediction</h2>
                <p><strong>Class:</strong> {predicted_class}</p>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
            """
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class InsightRequest(BaseModel):
    plastic_type: str

def clean_json_output(response_text: str) -> str:
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            return match.group(1)
        start = response_text.index('{')
        end = response_text.rindex('}') + 1
        return response_text[start:end]
    except Exception:
        return response_text.strip()

@app.post("/insights", tags=["Insights"])
async def get_insights(request: InsightRequest):
    prompt = f"""
    Provide a JSON object with the following structure only, without extra text or markdown formatting:

    {{
      "Plastic_name": "Full name of the plastic type",
      "Common_uses": ["list of common uses"],
      "Recycling_category": "Recycling number or symbol (e.g. #1)",
      "Environmental_impact": "Concise but detailed environmental impact",
      "Alternatives": ["list of sustainable alternatives"]
    }}

    The plastic type is: {request.plastic_type}
    """

    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=AIzaSyDqcZHYRtufGxHuy4RGrFKe05aIUL96E6s",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )
        response.raise_for_status()
        gemini_data = response.json()

        if "candidates" not in gemini_data:
            raise HTTPException(status_code=500, detail="Gemini API did not return a valid response")

        raw_output = gemini_data['candidates'][0]['content']['parts'][0]['text']
        cleaned = clean_json_output(raw_output)
        structured = json.loads(cleaned)

        return {
            "insight_html": f"""
                <h2>{structured.get("Plastic_name", "")}</h2>
                <h3>Common Uses</h3>
                <ul>{"".join(f"<li>{use}</li>" for use in structured.get("Common_uses", []))}</ul>
                <h3>Recycling Category</h3>
                <p>{structured.get("Recycling_category", "")}</p>
                <h3>Environmental Impact</h3>
                <p>{structured.get("Environmental_impact", "")}</p>
                <h3>Alternatives</h3>
                <ul>{"".join(f"<li>{alt}</li>" for alt in structured.get("Alternatives", []))}</ul>
            """
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Gemini API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Gemini response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


