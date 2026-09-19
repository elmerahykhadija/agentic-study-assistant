import sys
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END

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

# ==========================================
# 1. DÉFINITION DE LA MÉMOIRE (STATE)
# ==========================================
class GraphState(TypedDict, total=False):
    user_input: str
    input_type: str       # 'question', 'lien', ou 'document'
    use_ocr: bool
    progress_callback: any # Callable pour Streamlit UI
    context: str          # Les chunks récupérés
    evaluation: str       # 'correct', 'ambiguous', 'incorrect'
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
    result = extract_course_from_drive(state["user_input"], use_ocr=use_ocr, progress_callback=progress_cb)
    return {"final_answer": f"{result}\n(Vous pouvez maintenant poser des questions sur ce cours !)"}

def ingest_document_node(state: GraphState):
    print("\n▶️ NŒUD : Parsing de document/image direct")
    use_ocr = state.get("use_ocr", False)
    progress_cb = state.get("progress_callback", None)
    result = parse_local_document(state["user_input"], use_ocr=use_ocr, progress_callback=progress_cb)
    return {"final_answer": f"{result}\n(Vous pouvez maintenant poser des questions sur ce fichier !)"}

def retrieve_node(state: GraphState):
    print("\n▶️ NŒUD : Récupération du contexte (ChromaDB)")
    docs = retrieve_context(state["user_input"])
    context_str = "\n".join(docs) if docs else "Aucun document trouvé dans la base."
    return {"context": context_str}

def evaluate_node(state: GraphState):
    print("\n▶️ NŒUD : Évaluation CRAG")
    grade = evaluate_context(state["user_input"], state["context"])
    return {"evaluation": grade}

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
    answer = generate_standard_answer(
        state["user_input"], 
        state["context"], 
        state.get("chat_history", [])
    )
    return {"final_answer": answer}

def generate_visual_node(state: GraphState):
    print("\n▶️ NŒUD : Générateur Visuel (Schéma/PDF)")
    
    # On génère le diagramme
    answer = generate_visual_diagram(
        state["user_input"], 
        state["context"], 
        chat_history=state.get("chat_history", [])
    )
    
    return {"final_answer": answer}

def generate_pdf_node(state: GraphState):
    print("\n▶️ NŒUD : Générateur PDF")
    
    answer = generate_pdf_report(
        state["user_input"], 
        state["context"], 
        chat_history=state.get("chat_history", [])
    )
    
    return {"final_answer": answer}
# ==========================================
# 3. LA LOGIQUE DE ROUTAGE (EDGES)
# ==========================================
def orchestrator_router(state: GraphState):
    """Route l'entrée utilisateur (Lien Drive, Document ou Question)."""
    input_type = state.get("input_type")
    
    if input_type == "lien":
        return "ingest_drive"
    elif input_type == "document":
        return "ingest_document"
    else: # Si c'est "question" (ou qst)
        # Vérifier si l'utilisateur a inséré un lien ou pdf (base de données non vide)
        try:
            collection = get_chroma_collection()
            if collection.count() > 0:
                return "retrieve"
            else:
                return "web_search"
        except Exception:
            return "web_search"

def crag_router(state: GraphState):
    """Route selon la qualité du contexte (Correct ou Fallback Web)."""
    if state["evaluation"] == "correct":
        return "route_generator"
    return "web_search"

def generation_router(state: GraphState):
    """Route selon l'intention (Texte, Visuel, ou PDF)."""
    if state["generation_route"] == "pdf_document":
        return "generate_pdf"
    elif state["generation_route"] == "visual_document":
        return "generate_visual"
    return "generate_text"

# ==========================================
# 4. CONSTRUCTION DU GRAPHE
# ==========================================
workflow = StateGraph(GraphState)

# Ajout des nœuds
workflow.add_node("ingest_drive", ingest_drive_node)
workflow.add_node("ingest_document", ingest_document_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("route_generator", route_generator_node)
workflow.add_node("generate_text", generate_text_node)
workflow.add_node("generate_visual", generate_visual_node)
workflow.add_node("generate_pdf", generate_pdf_node)

# Définition des chemins
workflow.set_conditional_entry_point(
    orchestrator_router,
    {
        "ingest_drive": "ingest_drive", 
        "ingest_document": "ingest_document",
        "retrieve": "retrieve",
        "web_search": "web_search"
    }
)

# Les nœuds d'ingestion s'arrêtent après avoir vectorisé les données
workflow.add_edge("ingest_drive", END)
workflow.add_edge("ingest_document", END)

# Le parcours d'une question
workflow.add_edge("retrieve", "evaluate")
workflow.add_conditional_edges(
    "evaluate", crag_router,
    {"route_generator": "route_generator", "web_search": "web_search"}
)
workflow.add_edge("web_search", "route_generator")
workflow.add_conditional_edges(
    "route_generator", generation_router,
    {"generate_pdf": "generate_pdf", "generate_visual": "generate_visual", "generate_text": "generate_text"}
)
workflow.add_edge("generate_text", END)
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
    
    # Test d'une question classique qui traverse tout le graphe
    question = "Peux-tu m'expliquer le concept de RAG ?"
    inputs = {"user_input": question, "input_type": "question"}
    
    print(f"\n--- Lancement du test avec type : {inputs['input_type']} ---")
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"✅ {key} terminé.")