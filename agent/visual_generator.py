import os
import re
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv
import graphviz

# Obtenir le chemin absolu de la racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "diagram_output")

# Chargement de la clé API
load_dotenv(os.path.join(PROJECT_ROOT, "infra", ".env"))

def generate_visual_diagram(question: str, context: str, output_path: str = DEFAULT_OUTPUT_PATH, chat_history: list = None) -> str:
    """
    Génère un schéma d'architecture ou un logigramme basé sur le contexte.
    L'agent génère du code DOT (Graphviz) qui est ensuite compilé en PDF/PNG.
    """
    system_prompt = """
    You are a software architect and an expert in data visualization.
    Your mission is to answer the user's request by creating a conceptual diagram based ONLY on the provided CONTEXT.
    
    Strict Rules:
    1. You must generate EXCLUSIVELY Graphviz source code (DOT language).
    2. Enclose your code with ```dot at the beginning and ``` at the end.
    3. Use 'digraph G {' to start.
    4. Use shapes (shape=box, cylinder, etc.) and pastel colors to make the diagram aesthetic and professional.
    5. Do not add ANY explanatory text before or after the code.
    """
    
    generator_agent = Agent(
        model=Groq(id="openai/gpt-oss-120b"),
        description=system_prompt
    )
    
    prompt = f"CONTEXT:\n{context}\n\nREQUEST: {question}"
    
    if chat_history:
        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history])
        prompt = f"CHAT HISTORY:\n{history_str}\n\n{prompt}"
        
    print("🎨 Génération du diagramme en cours...")
    
    try:
        # 1. Demander le code au LLM
        response = generator_agent.run(prompt)
        content = response.content
        
        # 2. Extraire le code DOT des balises Markdown
        match = re.search(r'```dot\n(.*?)\n```', content, re.DOTALL)
        if match:
            dot_code = match.group(1)
        else:
            # Fallback si le LLM a oublié les balises
            dot_code = content.strip().replace('```', '')
            
        # 3. Sauvegarder et compiler avec Graphviz
        # S'assurer que le dossier existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        graph = graphviz.Source(dot_code)
        # Rend le fichier en PDF et PNG
        graph.render(output_path, format='png', cleanup=True)
        graph.render(output_path, format='pdf', cleanup=True)
        
        print(f"✅ Schéma généré avec succès : {output_path}.png et .pdf")
        return f"[GÉNÉRATION VISUELLE] Le diagramme a été généré avec succès ! Vous pouvez le consulter ici : {output_path}.pdf"
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération visuelle : {e}")
        return "Une erreur est survenue lors de la création du diagramme."

if __name__ == "__main__":
    # Test local
    q = "Fais-moi un schéma expliquant l'architecture RAG avec ses trois étapes principales."
    ctx = "Le RAG (Retrieval-Augmented Generation) fonctionne en 3 étapes : 1. L'ingestion des documents (PDFs) dans une base vectorielle. 2. La récupération (Retrieval) du contexte pertinent via une recherche de similarité. 3. La génération par le LLM qui lit le contexte pour formuler la réponse."
    
    rep = generate_visual_diagram(q, ctx)
    print("\n--- Résultat ---")
    print(rep)