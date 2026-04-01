from flask import Flask, request, jsonify
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from dotenv import load_dotenv
from flask_cors import CORS
import assemblyai as aai
import os
import json
import base64
import requests
import tempfile


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")

aai.settings.api_key = ASSEMBLY_API_KEY

model = init_chat_model(
    "google_genai:gemini-2.5-flash", 
    api_key = GOOGLE_API_KEY
)

checkpointer = InMemorySaver()

agent = create_agent(
    model = model, 
    tools = [], 
    checkpointer=checkpointer
)

current_subject = ""
question_count = 0
thread_id = "interview_session_1"
INTERVIEW_PROMPT = """You are Natalie, a friendly and conversational interviewer conducting a natural {subject} interview.

IMPORTANT GUIDELINES:
1. Ask exactly 5 questions total throughout the interview
2. Keep questions SHORT and CRISP (1-2 sentences maximum)
3. ALWAYS reference what the candidate ACTUALLY said in their previous answer - do NOT make up or assume their answers
4. Show genuine interest with brief acknowledgments based on their REAL responses
5. Adapt questions based on their ACTUAL responses - go deeper if they're strong, adjust if uncertain
6. Be warm and conversational but CONCISE
7. No lengthy explanations - just ask clear, direct questions

CRITICAL: Read the conversation history carefully. Only acknowledge what the candidate truly said, not what you think they might have said.

Keep it short, conversational, and adaptive!"""

FEEDBACK_PROMPT = """Based on our complete interview conversation, provide detailed feedback.
        IMPORTANT: You MUST respond with ONLY a valid JSON object. No other text before or after.
        Address the candidate directly using "you" and "your" (e.g., "You explained..." not "The candidate explained...").
        Respond with ONLY this JSON structure (no markdown, no code blocks, no extra text):
        {{
            "subject": "{subject}",
            "candidate_score": <1-5>,
            "feedback": "<detailed strengths with specific examples from their ACTUAL answers>",
            "areas_of_improvement": "<constructive suggestions based on gaps you noticed>"
        }}
        Be specific - reference ACTUAL things they said during the interview."""

app = Flask(__name__)
CORS(app, expose_headers=['X-Question-Number'])


def parse_feedback_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("Feedback response was not valid JSON")

def stream_audio(text):
    BASE_URL = "https://global.api.murf.ai/v1/speech/stream"
    payload = {
        "text": text,
        "voiceId": "en-US-Natalie",
        "voice": "natalie",
        "model": "FALCON",
        "multiNativeLocale": "en-US",
        "sampleRate": 24000,
        "format": "MP3",
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": MURF_API_KEY
    }
    response = requests.post(
        BASE_URL,
        headers=headers,
        data=json.dumps(payload),
        stream=True
    )
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            yield base64.b64encode(chunk).decode("utf-8") + "\n"


@app.route("/start-interview", methods=["POST"])
def start_interview():
    global current_subject, question_count, agent, checkpointer
    data = request.json
    current_subject = data.get("subject", "python")
    question_count = 1
    # thread_id = data.get("thread_id", "interview_session_1")

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer
    )

    formatted_prompt = INTERVIEW_PROMPT.format(subject=current_subject)
    config = {"configurable": {"thread_id" : thread_id} }
    response = agent.invoke({
        "messages": [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": f"Start the interview with a warm greeting and ask the first question about {current_subject}. Keep it SHORT (1-2 sentences)."}
        ]
    }, config=config)

    question = response["messages"][-1].content
    return jsonify({
        "success": True,
        "question": question,
        "question_number": question_count
    })

def speech_to_text(audio_path):
    config = aai.TranscriptionConfig(
        speech_models=["universal-3-pro", "universal-2"],
        language_detection=True
    )

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=config)

    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    return transcript.text if transcript.text else ""


@app.route("/submit-answer", methods=["POST"])
def submit_answer():
    global question_count

    # Get audio file
    audio_file = request.files['audio']
    current_answer_number = question_count

    # Save temp file
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".webm").name
    audio_file.save(temp_path)

    # Convert speech → text
    try:
        answer = speech_to_text(temp_path)
    finally:
        os.unlink(temp_path)

    if not answer:
        answer = "Empty Text Received..."

    print(f"Answer {current_answer_number}: {answer}")

    # Maintain conversation thread
    config = {"configurable": {"thread_id": thread_id}}

    if current_answer_number >= 5:
        return jsonify({
            "success": True,
            "answer": answer,
            "question_number": current_answer_number,
            "interview_completed": True,
            "message": "Interview completed"
        })

    question_count += 1

    # Build prompt for next question
    prompt = f"""
The candidate just answered question {current_answer_number}.

Their actual answer:
{answer}

Now:
1. Briefly acknowledge their REAL response (1 sentence)
2. Ask question {question_count} of 5 based on their answer
3. Keep total response under 3 sentences
4. If answer is weak, simplify next question
"""

    # Invoke agent
    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        config=config
    )

    # Extract next question
    question = response["messages"][-1].content

    print(f"[Question {question_count}] {question}")

    # Return response to frontend
    return jsonify({
        "success": True,
        "answer": answer,
        "question": question,
        "question_number": question_count,
        "interview_completed": False
    })

@app.route("/get-feedback", methods=["POST"])
def get_feedback():
    if question_count == 0 or not current_subject:
        return jsonify({
            "success": False,
            "error": "No interview session found"
        }), 400

    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke({
        "messages": [
        {
            "role": "user", 
            "content": (
                f"{FEEDBACK_PROMPT.format(subject=current_subject)}\n\n"
                f"Review our complete {current_subject} interview conversation and provide detailed feedback."
            )
        }
        ]
    }, config=config)
    text = response["messages"][-1].content
    print(f"\n[Feedback Generated]\n{text}\n")
    feedback = parse_feedback_response(text)
    return jsonify({
        "success": True,
        "feedback": feedback
    })

app.run(debug=True, port=5000)
