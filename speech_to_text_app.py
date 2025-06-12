import os
import json
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import speech
from google.oauth2 import service_account

# --- Basic Setup ---
app = Flask(__name__)
CORS(app)
load_dotenv()

from flask_sock import Sock
sock = Sock(app)

# --- Gemini Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    print("✅ Gemini Model configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Gemini: {e}")
    gemini_model = None

# --- Google Cloud Speech-to-Text Setup for Railway (JSON as env variable) ---
google_creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
google_creds_dict = json.loads(google_creds_json)
credentials = service_account.Credentials.from_service_account_info(google_creds_dict)

speech_client = speech.SpeechClient(credentials=credentials)


def get_gemini_extraction(transcript, source_lang):
    """Sends the final transcript to Gemini for translation and structured data extraction."""
    if not gemini_model:
        return {"error": "Gemini model not configured."}
    if not transcript.strip():
        return {"error": "Cannot process empty transcript."}

    source_language_full_name = "English" if source_lang == "en" else "Malayalam"

    default_structure = {
        "final_english_text": transcript,
        "extracted_terms": {
            "Medicine Names": [],
            "Dosage & Frequency": [],
            "Diseases / Conditions": [],
            "Symptoms": [],
            "Medical Procedures / Tests": [],
            "Duration": [],
            "Doctor's Instructions": []
        },
        "source_language": source_lang
    }

    smart_prompt = f"""
    You are an advanced medical transcription AI. Your input is raw text from a speech-to-text system.
    The source language of the text is {source_language_full_name}.

    Your tasks are:
    1. If the input is in Malayalam, translate it to high-quality medical English.
    2. If the input is in English, clean it up and normalize medical terminology.
    3. Extract key medical information into a structured JSON format.

    Raw Transcript Input:
    "{transcript}"

    Required JSON Output Format:
    {{
      "final_english_text": "The fully translated and normalized English text",
      "extracted_terms": {{
        "Medicine Names": ["...", "..."],
        "Dosage & Frequency": ["...", "..."],
        "Diseases / Conditions": ["...", "..."],
        "Symptoms": ["...", "..."],
        "Medical Procedures / Tests": ["...", "..."],
        "Duration": ["...", "..."],
        "Doctor's Instructions": ["...", "..."]
      }},
      "source_language": "{source_lang}"
    }}

    Important Rules:
    - Always output valid JSON. Do not include any text before or after the JSON object.
    - The "source_language" in the JSON must be "{source_lang}".
    - If no terms are found in a category, return an empty array for that key.
    - Ensure all keys from the example are present in your output.
    """
    try:
        response = gemini_model.generate_content(smart_prompt)
        cleaned_json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
        gemini_result = json.loads(cleaned_json_string)

        final_result = default_structure
        final_result["final_english_text"] = gemini_result.get("final_english_text", transcript)
        
        if "extracted_terms" in gemini_result and isinstance(gemini_result.get("extracted_terms"), dict):
            for category in final_result["extracted_terms"]:
                final_result["extracted_terms"][category] = gemini_result["extracted_terms"].get(category, [])

        return final_result

    except Exception as e:
        print(f"❌ Gemini processing error: {e}")
        error_response = default_structure
        error_response["error"] = "Failed to process text with Gemini."
        error_response["details"] = str(e)
        return error_response


@sock.route('/speech/<lang_code>')
def speech_socket(ws, lang_code):
    """Handles the WebSocket connection for live speech transcription."""
    print(f"🟢 Client connected for language: {lang_code}. Starting Google STT stream...")

    if lang_code == 'ml':
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ml-IN",
            enable_automatic_punctuation=True,
            model="latest_long"
        )
    else:
        lang_code = 'en'
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True,
            model="medical_dictation"
        )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    def request_generator(websocket):
        try:
            while True:
                message = websocket.receive()
                if message is None:
                    break
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        if data.get('type') == 'end_stream':
                            print("Generator: Received 'end_stream' signal. Ending audio stream.")
                            break
                    except json.JSONDecodeError:
                        continue
                else:
                    yield speech.StreamingRecognizeRequest(audio_content=message)
        except Exception as e:
            print(f"Generator error: {e}")

    try:
        responses = speech_client.streaming_recognize(
            config=streaming_config,
            requests=request_generator(ws)
        )
        final_transcript = ""
        for response in responses:
            if not ws.connected:
                break
            if not response.results:
                continue
            result = response.results[0]
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            ws.send(json.dumps({"type": "transcript", "is_final": result.is_final, "text": transcript}))
            if result.is_final:
                final_transcript += transcript + " "

        print(f"✅ Final Transcript for Gemini: {final_transcript}")
        if final_transcript.strip():
            gemini_result = get_gemini_extraction(final_transcript, lang_code)
            ws.send(json.dumps({"type": "entities", "data": gemini_result}))
    except Exception as e:
        print(f"❌ Error during streaming: {e}")
        try:
            ws.send(json.dumps({"type": "error", "message": f"Streaming Error: {e}"}))
        except:
            pass
    finally:
        print("🔴 Stream closed.")
        try:
            if ws.connected:
                ws.close()
        except:
            pass


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
