import gradio as gr
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer
import faiss
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import os
import PyPDF2
import re

hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# === 1. Define agent state ===
class AgentState(TypedDict):
    query: str
    context: str
    answer: str

# === 2. Load and chunk data ===
DATA_PATH = "data/mm7345a4-H.pdf"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def clean_text(text: str) -> str:
    # Remove extra whitespace and normalize text
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def chunk_text(text: str) -> List[str]:
    # Split text into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if current_length + len(sentence) > CHUNK_SIZE:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Load and process PDF
raw_text = extract_text_from_pdf(DATA_PATH)
cleaned_text = clean_text(raw_text)
chunks = chunk_text(cleaned_text)

# Create metadata for chunks
metadata = [{
    "source": "WHO Measles Fact Sheet",
    "page": "PDF Document",
    "id": i
} for i in range(len(chunks))]

# === 3. Build FAISS index ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks, convert_to_numpy=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# === 4. Load LLM ===
MODEL_ID = "UBC-NLP/NileChat-3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
model.eval()

# === 5. LangGraph Nodes ===
def retrieve_context(state: AgentState) -> dict:
    query = state["query"]
    q_vec = embedder.encode([query])
    D, I = index.search(q_vec, 3)  # Get top 3 most relevant chunks
    retrieved_chunks = [f"[Source: {metadata[i]['source']}] {chunks[i].strip()}" for i in I[0]]
    return {"context": "\n\n".join(retrieved_chunks)}

def generate_answer(state: AgentState) -> dict:
    prompt = (
        f"You are a medical expert specialized in measles.\n"
        f"Use the following context to answer the question clearly and accurately.\n"
        f"If the context doesn't contain enough information to answer the question, say so.\n\n"
        f"Context:\n{state['context']}\n\n"
        f"Question: {state['query']}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer.strip()}

# === 6. Build LangGraph ===
graph_builder = StateGraph(AgentState)
graph_builder.add_node("retrieve_context", retrieve_context)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_edge(START, "retrieve_context")
graph_builder.add_edge("retrieve_context", "generate_answer")
graph_builder.add_edge("generate_answer", END)
graph_builder.set_entry_point("retrieve_context")
graph = graph_builder.compile()

# === 7. Gradio Interface ===
def run_agent(query: str) -> str:
    state = AgentState(query=query, context="", answer="")
    final_state = graph.invoke(state)
    return final_state["answer"]

# Launch Gradio
demo = gr.Interface(
    fn=run_agent,
    inputs=gr.Textbox(lines=2, placeholder="Ask something about measles..."),
    outputs=gr.Textbox(label="Answer"),
    title="Measles Expert Agent",
    description="Ask any medical question related to measles, and get an answer powered by LangGraph, FAISS, and NileChat-3B.",
)

if __name__ == "__main__":
    demo.launch()
