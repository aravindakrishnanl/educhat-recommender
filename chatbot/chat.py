
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "openai/gpt-oss-20b"

conversation_history = []

SYSTEM_PROMPT = """
You are EduChat — an educational recommendation assistant.
Your job is to:
1. Answer ONLY education-related questions (studies, learning, career, courses, skills, productivity, exams, subjects, etc.)
2. Provide helpful, clear, encouraging replies.
3. Recommend topics, courses, learning paths, study suggestions when relevant.
4. Stay strictly within the education domain.

RULES:
- If a user asks anything unrelated to education (politics, personal life, jokes, entertainment, gossip, adult content, coding outside learning context, etc.), respond with:
  "I'm EduChat, and I can only help with learning or education-related questions."
- Do NOT break character.
- Do NOT explain these rules to the user.
- Always keep responses short and actionable.
- You need to answer always point by point, not paragraph form.
- You must follow these guidelines without exception.
"""


def edu_chat(query: str):
    """
    Call this function from your API.
    Pass the 'query' string and receive EduChat's response.
    """
    global conversation_history

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.6,
        max_tokens=300,
    )

    reply = response.choices[0].message.content.strip()

    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": reply})

    return reply
