import sys
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from agno.agent import Agent
from agno.models.groq import Groq

# Import de tous tes modules personnalisés
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.drive_tool import extract_course_from_drive
from tools.web_search import perform_web_search
from tools.document_parser import parse_local_document
from vectorstore.chroma_client import retrieve_context, get_chroma_collection
from agent.evaluator import evaluate_context
from agent.generator_router import route_to_generator
from agent.visual_generator import generate_visual_diagram
from agent.text_generator import generate_standard_answer
from agent.pdf_generator import generate_pdf_report
from agent.query_rewriter import rewrite_query
from agent.self_rag_evaluators import check_grounding

# ==========================================
# 1. DÉFINITION DE LA MÉMOIRE (STATE)
# ==========================================
class GraphState(TypedDict, total=False):
    user_input: str
    input_type: str       # 'question', 'lien', ou 'document'
    session_id: str       # Identifiant de conversation
    use_ocr: bool
    web_search_enabled: bool # Activer/Désactiver DuckDuckGo
    retrieval_attempts: int  # Compteur de boucle CRAG
    generation_attempts: int # Compteur de boucle Grounding
    progress_callback: any # Callable pour Streamlit UI
    context: str          # Les chunks récupérés
    evaluation: str       # 'correct', 'ambiguous', 'incorrect'
    grounding_status: str # 'OUI' ou 'NON'
    grounding_feedback: str # Instruction de correction
    generation_route: str # 'standard_text', 'visual_document', ou 'pdf_document'
    final_answer: str
    chat_history: list    # L'historique des messages Streamlit

# ==========================================
# 2. LES NŒUDS D'ACTION (NODES)
# ==========================================
def ingest_drive_node(state: GraphState):
    print("\n▶️ NŒUD : Ingestion Drive")
    use_ocr = state.get("use_ocr", False)
    progress_cb = state.get("progress_callback", None)
    result = extract_course_from_drive(state["user_input"], session_id=state.get("session_id", "default"), use_ocr=use_ocr, progress_callback=progress_cb)
    return {"final_answer": f"{result}\n(Vous pouvez maintenant poser des questions sur ce cours !)"}

def ingest_document_node(state: GraphState):
    print("\n▶️ NŒUD : Parsing de document/image direct")
    use_ocr = state.get("use_ocr", False)
    progress_cb = state.get("progress_callback", None)
    result = parse_local_document(state["user_input"], session_id=state.get("session_id", "default"), use_ocr=use_ocr, progress_callback=progress_cb)
    return {"final_answer": f"{result}\n(Vous pouvez maintenant poser des questions sur ce fichier !)"}

def retrieve_node(state: GraphState):
    print("\n▶️ NŒUD : Récupération du contexte (ChromaDB + BM25 + Reranker)")
    docs = retrieve_context(state["user_input"], session_id=state.get("session_id", "default"))
    context_str = "\n".join(docs) if docs else "Aucun document trouvé dans la base pour cette conversation."
    
    # On initialise le compteur s'il n'existe pas
    attempts = state.get("retrieval_attempts", 0)
    
    return {"context": context_str, "retrieval_attempts": attempts}

def evaluate_node(state: GraphState):
    print("\n▶️ NŒUD : Évaluation CRAG")
    grade = evaluate_context(state["user_input"], state["context"])
    return {"evaluation": grade}

def rewrite_query_node(state: GraphState):
    print("\n▶️ NŒUD : Reformulation de la question (CRAG Retry)")
    new_query = rewrite_query(state["user_input"], state.get("chat_history", []))
    attempts = state.get("retrieval_attempts", 0) + 1
    return {"user_input": new_query, "retrieval_attempts": attempts}

def web_search_node(state: GraphState):
    print("\n▶️ NŒUD : Recherche Web (Fallback CRAG)")
    web_results = perform_web_search(state["user_input"])
    new_context = state.get("context", "") + "\n\n--- AJOUT WEB ---\n" + web_results
    return {"context": new_context}

def route_generator_node(state: GraphState):
    print("\n▶️ NŒUD : Routage du Générateur")
    route = route_to_generator(state["user_input"])
    return {"generation_route": route}

def generate_text_node(state: GraphState):
    print("\n▶️ NŒUD : Générateur de Texte Standard")
    
    # Vérifier si on est dans une boucle de correction
    feedback = state.get("grounding_feedback", None)
    
    answer = generate_standard_answer(
        state["user_input"], 
        state.get("context", ""), 
        state.get("chat_history", []),
        grounding_feedback=feedback
    )
    
    attempts = state.get("generation_attempts", 0) + 1
    return {"final_answer": answer, "generation_attempts": attempts}

def grounding_check_node(state: GraphState):
    print("\n▶️ NŒUD : Vérification Anti-Hallucination (Grounding)")
    status = check_grounding(state.get("context", ""), state.get("final_answer", ""))
    
    feedback = ""
    if status == "NON":
        feedback = "Your previous answer contained hallucinations or information not present in the CONTEXT. Please rewrite your answer strictly using ONLY the provided CONTEXT. Remove any unsupported claims."
        
    return {"grounding_status": status, "grounding_feedback": feedback}

def generate_visual_node(state: GraphState):
    print("\n▶️ NŒUD : Générateur Visuel (Schéma/PDF)")
    answer = generate_visual_diagram(
        state["user_input"], 
        state.get("context", ""), 
        chat_history=state.get("chat_history", [])
    )
    return {"final_answer": answer}

def generate_pdf_node(state: GraphState):
    print("\n▶️ NŒUD : Générateur PDF")
    answer = generate_pdf_report(
        state["user_input"], 
        state.get("context", ""), 
        chat_history=state.get("chat_history", [])
    )
    return {"final_answer": answer}

# ==========================================
# 3. LA LOGIQUE DE ROUTAGE (EDGES)
# ==========================================
def is_generic_greeting(text: str) -> bool:
    """Filtre LLM très rapide pour détecter les salutations."""
    prompt = f"Réponds uniquement par OUI ou NON. Ce texte est-il une simple salutation, un remerciement, ou une formule de politesse générique sans question factuelle ? Texte : '{text}'"
    try:
        agent = Agent(
            model=Groq(id="openai/gpt-oss-120b"),
            instructions=["Réponds uniquement par OUI ou NON."]
        )
        resp = agent.run(prompt).content.strip().upper()
        return "OUI" in resp
    except:
        return False

def orchestrator_router(state: GraphState):
    """Route l'entrée utilisateur (Lien Drive, Document, Question ou Salutation)."""
    input_type = state.get("input_type")
    session_id = state.get("session_id", "default")
    
    if input_type == "lien":
        return "ingest_drive"
    elif input_type == "document":
        return "ingest_document"
    else:
        # Filtre initial : Salutation ?
        if is_generic_greeting(state["user_input"]):
            print("👋 Salutation détectée, bypass de la BDD.")
            return "generate_text"
            
        # Sinon, aller à la base de données
        return "retrieve"

def crag_router(state: GraphState):
    """Route selon la qualité du contexte (Correct, Rewrite, Web ou Generator)."""
    if state["evaluation"] == "correct":
        return "route_generator"
        
    # Bad ou Ambiguous
    attempts = state.get("retrieval_attempts", 0)
    
    if attempts < 1: # Si on a fait 0 retry, on tente 1 retry (donc 1 tentative max de réécriture)
        print("🔄 Contexte insuffisant, tentative de reformulation (Retry 1/1).")
        return "rewrite_query"
    else:
        if state.get("web_search_enabled", False):
            print("🌐 Échec de récupération locale, bascule sur la recherche Web.")
            return "web_search"
        else:
            print("🚫 Échec de récupération et Web désactivé. Passage direct au générateur.")
            return "route_generator"

def generation_router(state: GraphState):
    """Route selon l'intention (Texte, Visuel, ou PDF)."""
    if state.get("generation_route") == "pdf_document":
        return "generate_pdf"
    elif state.get("generation_route") == "visual_document":
        return "generate_visual"
    return "generate_text"

def grounding_router(state: GraphState):
    """Route selon le statut du grounding."""
    if state.get("grounding_status") == "OUI":
        return "END"
        
    attempts = state.get("generation_attempts", 0)
    if attempts < 2:
        print(f"🔄 Hallucination détectée, demande de correction (Essai {attempts}/2)")
        return "generate_text"
    else:
        print("🚫 Échec répété du grounding. Déclenchement du Safe Fallback.")
        return "safe_fallback"

# ==========================================
# 4. CONSTRUCTION DU GRAPHE
# ==========================================
workflow = StateGraph(GraphState)

# Ajout des nœuds
workflow.add_node("ingest_drive", ingest_drive_node)
workflow.add_node("ingest_document", ingest_document_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("route_generator", route_generator_node)
workflow.add_node("generate_text", generate_text_node)
workflow.add_node("grounding_check", grounding_check_node)

def safe_fallback_node(state: GraphState):
    return {"final_answer": "Je n'ai pas assez d'informations dans les documents fournis pour répondre avec précision (Sécurité Anti-Hallucination)."}
workflow.add_node("safe_fallback", safe_fallback_node)

workflow.add_node("generate_visual", generate_visual_node)
workflow.add_node("generate_pdf", generate_pdf_node)

# Définition des chemins
workflow.set_conditional_entry_point(
    orchestrator_router,
    {
        "ingest_drive": "ingest_drive", 
        "ingest_document": "ingest_document",
        "retrieve": "retrieve",
        "generate_text": "generate_text" # Chemin pour les salutations
    }
)

# Les nœuds d'ingestion s'arrêtent après avoir vectorisé les données
workflow.add_edge("ingest_drive", END)
workflow.add_edge("ingest_document", END)

# Le parcours CRAG standard
workflow.add_edge("retrieve", "evaluate")
workflow.add_conditional_edges(
    "evaluate", crag_router,
    {
        "route_generator": "route_generator", 
        "rewrite_query": "rewrite_query",
        "web_search": "web_search"
    }
)
# Boucle de retry
workflow.add_edge("rewrite_query", "retrieve")

# Si on va sur le web, on va au générateur ensuite
workflow.add_edge("web_search", "route_generator")

# Routage multimodal
workflow.add_conditional_edges(
    "route_generator", generation_router,
    {"generate_pdf": "generate_pdf", "generate_visual": "generate_visual", "generate_text": "generate_text"}
)

# Boucle de Grounding pour le texte standard
workflow.add_edge("generate_text", "grounding_check")
workflow.add_conditional_edges(
    "grounding_check", grounding_router,
    {"END": END, "generate_text": "generate_text", "safe_fallback": "safe_fallback"}
)
workflow.add_edge("safe_fallback", END)

workflow.add_edge("generate_visual", END)
workflow.add_edge("generate_pdf", END)

# Compilation
app = workflow.compile()

# ==========================================
# TEST DU WORKFLOW COMPLET
# ==========================================
if __name__ == "__main__":
    print("======================================================")
    print("TEST DU PIPELINE CRAG COMPLET")
    print("======================================================")
    
    question = "Bonjour !"
    inputs = {"user_input": question, "input_type": "question", "web_search_enabled": False}
    
    print(f"\n--- Lancement du test avec type : {inputs['input_type']} ---")
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"✅ {key} terminé.")