import os
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv('../infra/.env')

def route_to_generator(user_query: str) -> str:
    """
    Analyse l'intention de l'utilisateur pour déterminer le type de réponse attendu.
    Retourne 'standard_text', 'visual_document' ou 'pdf_document'.
    """
    router_system_prompt = """You are an intent routing classifier. Your sole purpose is to analyze the user's query and route it to the correct generation pipeline.

Routing Rules:
- If the user explicitly asks to generate a PDF document, PDF report, or PDF summary -> Output EXACTLY "pdf_document".
- If the user explicitly asks to generate a diagram, schema, visual representation, roadmap, or flowchart (but NOT a PDF) -> Output EXACTLY "visual_document".
- For any other query (e.g., standard questions, summaries, definitions, explanations, standard text generation) -> Output EXACTLY "standard_text".

CRITICAL INSTRUCTIONS:
- You must output ONLY one of the three exact strings: "pdf_document", "visual_document", or "standard_text".
- Do NOT output any other text, punctuation, explanations, or conversational filler.
- If you are unsure, default to "standard_text".
"""
    
    # Utilisation d'un modèle léger et rapide pour le routage
    router_agent = Agent(
        model=Groq(id="openai/gpt-oss-120b"),
        description=router_system_prompt,
        instructions=["Réponds uniquement par 'pdf_document', 'visual_document' ou 'standard_text'."]
    )
    
    print(f"🚦 Analyse de l'intention pour : '{user_query}'...")
    
    try:
        try:
            response = router_agent.run(user_query)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower() or "expire" in str(e).lower() or "insufficient" in str(e).lower():
                print("⚠️ Limite de tokens atteinte, bascule sur l'API 2...")
                fallback_agent = Agent(
                    model=Groq(id="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY2")),
                    description=router_system_prompt,
                    instructions=["Réponds uniquement par 'pdf_document', 'visual_document' ou 'standard_text'."]
                )
                response = fallback_agent.run(user_query)
            else:
                raise e
                
        route = response.content.strip().lower()
        
        # Sécurité : Fallback sur le texte standard si le LLM hallucine son format
        if route not in ["pdf_document", "visual_document", "standard_text"]:
            print(f"⚠️ Réponse inattendue du routeur ('{route}'). Bascule par défaut sur 'standard_text'.")
            return "standard_text"
            
        print(f"✅ Route choisie : {route.upper()}")
        return route
        
    except Exception as e:
        print(f"❌ Erreur lors du routage : {e}")
        return "standard_text" # Fallback en cas d'erreur API

if __name__ == "__main__":
    # Tests locaux
    print("--- Test 1 (Attendu : STANDARD_TEXT) ---")
    route_to_generator("Peux-tu m'expliquer le concept de RAG ?")
    
    print("\n--- Test 2 (Attendu : VISUAL_DOCUMENT) ---")
    route_to_generator("Génère un schéma de l'architecture de cette base de données en PDF.")