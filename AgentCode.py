import gradio as gr
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import json
import os
import PyPDF2
import re
import redis
import hashlib
import uuid
from together import Together
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === Redis Setup ===
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
CACHE_TTL_SECONDS = 3600

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    redis_client.ping()
    print(f"[CACHE] Redis connected ✅")
except redis.RedisError:
    redis_client = None
    print(f"[CACHE] Redis unavailable, fallback to local cache ⚠️")

local_cache = {}

class AgentState(TypedDict):
    query: str
    context: str
    answer: Union[str, List[dict]]

DATA_PATH = "data/mm7345a4-H.pdf"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, chunk = [], []
    length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if length + len(sentence) > CHUNK_SIZE:
            chunks.append(' '.join(chunk))
            overlap = chunk[-CHUNK_OVERLAP:] if len(chunk) > CHUNK_OVERLAP else chunk
            chunk = overlap + [sentence]
            length = sum(len(s) for s in chunk)
        else:
            chunk.append(sentence)
            length += len(sentence)
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

# Check if PDF file exists
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] PDF file not found at {DATA_PATH}")
    print("Please ensure the PDF file is in the data/ directory")
    chunks = []
    metadata = []
else:
    raw_text = extract_text_from_pdf(DATA_PATH)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    metadata = [{"source": "WHO Measles Fact Sheet", "id": i} for i in range(len(chunks))]

embedder = SentenceTransformer("sentence-transformers/LaBSE")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embeddings = embedder.encode(chunks, convert_to_numpy=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Check for Together API key
together_api_key = os.environ.get("TOGETHER_API_KEY")
if not together_api_key:
    print("[ERROR] TOGETHER_API_KEY environment variable not set")
    print("Please set your Together AI API key: export TOGETHER_API_KEY=your_key_here")
    together_client = None
else:
    together_client = Together(api_key=together_api_key)

def compute_cache_key(query: str, mode: str) -> str:
    return hashlib.sha256(f"{mode}:{query}".encode("utf-8")).hexdigest()

def retrieve_context(state: AgentState) -> dict:
    query = state["query"]
    q_vec = embedder.encode([query])
    D, I = index.search(q_vec, 8)

    top_chunks = [chunks[i] for i in I[0]]
    pairs = [[query, chunk] for chunk in top_chunks]
    scores = reranker.predict(pairs)

    scored_chunks = sorted(zip(scores, top_chunks, I[0]), key=lambda x: x[0], reverse=True)[:3]
    context = ""
    for _, chunk, idx in scored_chunks:
        context += f"[Source: {metadata[idx]['source']}] {chunk.strip()[:500]}\n\n"

    print(f"[DEBUG] Retrieved context for query '{query}':\n{context}")
    return {"context": context}

def get_history(session_id: str) -> List[str]:
    if redis_client:
        key = f"history:{session_id}"
        raw = redis_client.get(key)
        if raw:
            entries = json.loads(raw)
            return [f"User: {entry['query']}\nAssistant: {entry['answer']}" for entry in entries][-4:]
    return []

def save_history(session_id: str, query: str, answer: Union[str, List[dict]]):
    entry = {"query": query, "answer": answer}
    if redis_client:
        key = f"history:{session_id}"
        raw = redis_client.get(key)
        data = json.loads(raw) if raw else []
        data.append(entry)
        if len(data) > 20:
            data = data[-20:]
        redis_client.set(key, json.dumps(data), ex=86400)
    else:
        file = f"./history_{session_id}.json"
        try:
            with open(file, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []
        data.append(entry)
        if len(data) > 20:
            data = data[-20:]
        with open(file, "w") as f:
            json.dump(data, f, indent=2)

def together_chat_completion(context_messages: List[str], user_query: str) -> str:
    if not together_client:
        return "Error: Together AI client not initialized. Please check your API key."
    
    # Build conversation history
    messages = []
    
    # Add system message
    system_message = {
        "role": "system",
        "content": "You are a highly reliable and factual health assistant specialized in measles-related public information. You operate inside a retrieval-augmented generation (RAG) system and must always ground your responses strictly on the documents retrieved by your tools.\n\nAlways:\n\nSummarize clearly and accurately, using only the retrieved content.\n\nCite the relevant source or section if needed (by title or identifier).\n\nUse simple, understandable language for the general public (no medical jargon unless explained).\n\nIf the information is not found, say so clearly and do not hallucinate or assume anything.\n\nYour main goals:\n\nHelp users understand symptoms, vaccines, outbreak news, and prevention measures.\n\nWarn users about misinformation and guide them to evidence-based practices.\n\nProvide trusted, actionable, and verified insights without making medical diagnoses.\n\nAlways maintain a neutral, reassuring, and educational tone. You are not a doctor, but a public-facing, AI-powered explainer focused on helping people stay safe and informed about measles."
    }
    messages.append(system_message)
    
    # Add conversation history
    for msg in context_messages[-4:]:
        if msg.startswith("User: "):
            messages.append({"role": "user", "content": msg[6:]})
        elif msg.startswith("Assistant: "):
            messages.append({"role": "assistant", "content": msg[11:]})
    
    # Add current user query
    messages.append({"role": "user", "content": user_query})
    
    output = ""
    try:
        response = together_client.chat.completions.create(
            model="google/gemma-3n-E4B-it",
            messages=messages,
            stream=True
        )
        for token in response:
            if hasattr(token, 'choices') and token.choices[0].delta.content:
                output += token.choices[0].delta.content
    except Exception as e:
        print(f"[GEN ERROR] {e}")
        return "Error during generation."
    return output.strip()

def generate_answer(state: AgentState) -> dict:
    session_history = get_history(session_id)
    
    # Build conversation history for context
    context_messages = []
    for msg in session_history[-4:]:
        context_messages.append(msg)
    
    # Add context information to the user query
    enhanced_query = f"Context:\n{state['context']}\n\nUser question: {state['query']}"
    
    response = together_chat_completion(context_messages, enhanced_query)
    return {"answer": response}

def generate_quiz(state: AgentState) -> dict:
    if not together_client:
        return {"answer": [{"question": "Error: Together AI client not initialized. Please check your API key.", "choices": [], "answer": ""}]}
    
    messages = [
        {
            "role": "system",
            "content": "You are a highly reliable and factual health assistant specialized in measles-related public information. You operate inside a retrieval-augmented generation (RAG) system and must always ground your responses strictly on the documents retrieved by your tools.\n\nAlways:\n\nSummarize clearly and accurately, using only the retrieved content.\n\nCite the relevant source or section if needed (by title or identifier).\n\nUse simple, understandable language for the general public (no medical jargon unless explained).\n\nIf the information is not found, say so clearly and do not hallucinate or assume anything.\n\nYour main goals:\n\nHelp users understand symptoms, vaccines, outbreak news, and prevention measures.\n\nWarn users about misinformation and guide them to evidence-based practices.\n\nProvide trusted, actionable, and verified insights without making medical diagnoses.\n\nAlways maintain a neutral, reassuring, and educational tone. You are not a doctor, but a public-facing, AI-powered explainer focused on helping people stay safe and informed about measles."
        },
        {
            "role": "user",
            "content": f"Generate 3 multiple-choice questions from this medical context. Each should have 4 choices (A-D) and the correct answer marked.\n\nContext:\n{state['context']}\n\nOutput format:\n[{{\"question\": ..., \"choices\": [\"A...\", \"B...\", ...], \"answer\": \"A\"}}]"
        }
    ]
    
    output = ""
    try:
        response = together_client.chat.completions.create(
            model="google/gemma-3n-E4B-it",
            messages=messages,
            stream=True
        )
        for token in response:
            if hasattr(token, 'choices') and token.choices[0].delta.content:
                output += token.choices[0].delta.content
    except Exception as e:
        print(f"[QUIZ ERROR] {e}")
        return {"answer": [{"question": "Error generating quiz", "choices": [], "answer": ""}]}
    
    try:
        start, end = output.find("["), output.rfind("]") + 1
        quiz = json.loads(output[start:end].replace("'", '"'))
    except Exception as e:
        print(f"[QUIZ PARSE ERROR] {e}")
        quiz = [{"question": "Error parsing quiz", "choices": [], "answer": ""}]
    return {"answer": quiz}

graph_builder_qa = StateGraph(AgentState)
graph_builder_qa.add_node("retrieve_context", retrieve_context)
graph_builder_qa.add_node("generate_answer", generate_answer)
graph_builder_qa.add_edge(START, "retrieve_context")
graph_builder_qa.add_edge("retrieve_context", "generate_answer")
graph_builder_qa.add_edge("generate_answer", END)
graph_builder_qa.set_entry_point("retrieve_context")
graph_qa = graph_builder_qa.compile()

graph_builder_quiz = StateGraph(AgentState)
graph_builder_quiz.add_node("retrieve_context", retrieve_context)
graph_builder_quiz.add_node("generate_quiz", generate_quiz)
graph_builder_quiz.add_edge(START, "retrieve_context")
graph_builder_quiz.add_edge("retrieve_context", "generate_quiz")
graph_builder_quiz.add_edge("generate_quiz", END)
graph_builder_quiz.set_entry_point("retrieve_context")
graph_quiz = graph_builder_quiz.compile()

session_id = str(uuid.uuid4())

def run_agent_with_memory(query: str, mode: str) -> Union[str, List[dict]]:
    key = compute_cache_key(query, mode)
    if redis_client:
        cached = redis_client.get(key)
        if cached:
            print(f"[CACHE HIT] Redis: {query}")
            return json.loads(cached)
    if key in local_cache:
        print(f"[CACHE HIT] Local: {query}")
        return local_cache[key]
    print(f"[CACHE MISS] {query}")
    state = AgentState(query=query, context="", answer="")
    final_state = graph_quiz.invoke(state) if mode == "quiz" else graph_qa.invoke(state)
    result = final_state["answer"]
    save_history(session_id, query, result)
    if redis_client:
        redis_client.set(key, json.dumps(result), ex=CACHE_TTL_SECONDS)
    else:
        local_cache[key] = result
    return result

def format_output(result):
    if isinstance(result, list):
        output = ""
        for i, q in enumerate(result):
            output += f"**Q{i+1}: {q['question']}**\n"
            for idx, choice in enumerate(q['choices']):
                output += f"- {chr(65+idx)}. {choice}\n"
            output += f"✅ **Answer**: {q['answer']}\n\n"
        return output
    return result

demo = gr.Interface(
    fn=lambda query, mode: format_output(run_agent_with_memory(query, mode)),
    inputs=[
        gr.Textbox(label="Query"),
        gr.Dropdown(choices=["answer", "quiz"], value="answer", label="Mode")
    ],
    outputs=gr.Markdown(label="Output"),
    title="🧠 Measles Assistant + Quiz Generator (Together AI)",
    description="Answers medical questions based on WHO data. RAG-powered with Gemma-3n-E4B-it, FAISS, reranker, and Redis."
)

if __name__ == "__main__":
    demo.launch(debug=True)
