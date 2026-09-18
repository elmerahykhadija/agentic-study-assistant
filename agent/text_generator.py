import os
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Chargement de la clé API
load_dotenv('../infra/.env')

def generate_standard_answer(question: str, context: str) -> str:
    """
    Prend le contexte (approuvé par le CRAG ou issu du Web) et génère la réponse finale.
    """
    system_prompt = """
    You are a clear and precise pedagogical study assistant.
    Your mission is to answer the user's QUESTION based ONLY on the provided CONTEXT.
    
    Writing Rules:
    1. Pedagogy: Explain concepts in a simple and accessible way, like a supportive tutor.
    2. Synthesis: If the context contains the mention "AJOUT WEB", synthesize the information clearly and structurally.
    3. Formatting: Use Markdown (bullet points, bold text, line breaks) to make the text easy to read.
    4. Immersion: Never mention the background process (do not say "Based on the context", "Following my research", etc.). Answer directly.
    5. Reliability: If the context does not contain enough information to answer the question, admit it honestly without inventing an answer.
    """
    
    # Llama 3 via Groq pour la génération de la réponse
    generator_agent = Agent(
        model=Groq(id="openai/gpt-oss-120b"),
        description=system_prompt,
    )
    
    prompt = f"CONTEXTE:\n{context}\n\nQUESTION: {question}"
    
    print("💬 Génération de la réponse textuelle en cours...")
    
    try:
        response = generator_agent.run(prompt)
        return response.content
    except Exception as e:
        print(f"❌ Erreur lors de la génération : {e}")
        return "Une erreur est survenue lors de la formulation de la réponse."

if __name__ == "__main__":
    # Test local
    q = "Qu'est-ce que le Big Data ?"
    ctx = "Le Big Data fait référence à des ensembles de données massifs. Les 3V (Volume, Vélocité, Variété) sont souvent utilisés pour le définir."
    
    rep = generate_standard_answer(q, ctx)
    print("\n--- Réponse Générée ---")
    print(rep)