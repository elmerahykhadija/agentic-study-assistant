import os
import sys
import tempfile
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.graph import app as langgraph_app
from vectorstore.chroma_client import clear_database

app = FastAPI(title="Agentic Study Assistant API", description="API pour le pipeline CRAG")

# Autoriser les requêtes CORS depuis l'application React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À configurer plus strictement en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount Static Directories for Downloads ---
pdf_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'pdf_output'))
diagram_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'diagram_output'))

os.makedirs(pdf_output_dir, exist_ok=True)
os.makedirs(diagram_output_dir, exist_ok=True)

app.mount("/api/download/pdf", StaticFiles(directory=pdf_output_dir), name="pdf_output")
app.mount("/api/download/diagram", StaticFiles(directory=diagram_output_dir), name="diagram_output")

# --- Modèles de données (Pydantic) ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    chat_history: List[ChatMessage] = []
    web_search_enabled: bool = False

class DriveIngestRequest(BaseModel):
    session_id: str
    drive_url: str

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Agentic Study Assistant API is running"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    inputs = {
        "user_input": request.message,
        "input_type": "question",
        "session_id": request.session_id,
        "web_search_enabled": request.web_search_enabled,
        "chat_history": [{"role": m.role, "content": m.content} for m in request.chat_history]
    }
    
    final_response = "Une erreur est survenue lors de la réflexion de l'agent."
    
    try:
        # Parcourir les événements générés par le graphe
        for output in langgraph_app.stream(inputs):
            for key, value in output.items():
                if "final_answer" in value:
                    final_response = value["final_answer"]
                    
        return {"response": final_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/drive")
async def ingest_drive(request: DriveIngestRequest):
    inputs = {
        "user_input": request.drive_url,
        "input_type": "lien",
        "session_id": request.session_id
    }
    
    try:
        final_msg = ""
        for output in langgraph_app.stream(inputs):
            for key, value in output.items():
                if "final_answer" in value:
                    final_msg = value["final_answer"]
        return {"message": final_msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/file")
async def ingest_file(
    file: UploadFile = File(...), 
    session_id: str = Form(...)
):
    # Enregistrer temporairement le fichier
    suffix = f".{file.filename.split('.')[-1]}" if '.' in file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
        
    inputs = {
        "user_input": tmp_file_path,
        "input_type": "document",
        "session_id": session_id
    }
    
    try:
        final_msg = ""
        for output in langgraph_app.stream(inputs):
            for key, value in output.items():
                if "final_answer" in value:
                    final_msg = value["final_answer"]
        return {"message": final_msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@app.post("/api/clear")
async def clear_db():
    try:
        clear_database()
        return {"message": "Base de connaissances purgée avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
