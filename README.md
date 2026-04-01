# AI Voice-Based Interview Assistant

## 🚀 Overview
This project is an AI-powered voice-based interview assistant that simulates real interview scenarios.  
It uses speech-to-text, large language models, and text-to-speech to create a fully interactive experience.

---

## ✨ Features
- 🎤 Voice-based interview interaction
- 🧠 Dynamic question generation using AI
- 🔁 Multi-turn conversation with memory
- 🔊 Text-to-speech responses (Murf AI)
- 📝 Speech-to-text conversion (AssemblyAI)
- 🎯 Adaptive questions based on user answers

---

## 🛠 Tech Stack
- **Backend:** Flask (Python)
- **LLM:** Google Gemini (via LangChain)
- **Speech-to-Text:** AssemblyAI
- **Text-to-Speech:** Murf AI
- **Memory:** LangGraph

---

## 🔄 Workflow
1. User starts interview with a subject
2. AI asks the first question
3. User answers using voice input
4. Audio is converted to text
5. AI analyzes the answer
6. Next question is generated
7. Process repeats for 5 questions

---

## 📡 API Endpoints

### ▶️ Start Interview
**POST** `/start-interview`

**Request:**
```json
{
  "subject": "python"
}
