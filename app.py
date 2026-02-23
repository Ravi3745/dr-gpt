import os
import uuid
from collections import defaultdict, deque

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from src.prompt import system_prompt

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "dr-gpt")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# Remember last 6 turns (user+ai) per session
MAX_TURNS = 6
chat_store = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))  # 12 messages

# ---------- Vector Store ----------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

retriever = docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1024)

# Add chat_history placeholder so model sees previous turns
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),          # should include {context}
    ("placeholder", "{chat_history}"),  # <-- memory injected here
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def get_session_id() -> str:
    """One id per browser session; used as key in chat_store."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

@app.get("/")
def home():
    get_session_id()
    return render_template("chat.html")

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()
    if not user_input:
        return jsonify({"answer": "Please ask a question."}), 400

    sid = get_session_id()
    history = chat_store[sid]

    try:
        result = rag_chain.invoke({
            "input": user_input,
            "chat_history": list(history),
        })

        answer = result["answer"]

        # Save last 6 turns (auto-trim by deque maxlen)
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=answer))

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Server error: {str(e)}"}), 500

@app.post("/reset")
def reset():
    """ clear memory for this browser session."""
    sid = get_session_id()
    chat_store.pop(sid, None)
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True, port=5000)