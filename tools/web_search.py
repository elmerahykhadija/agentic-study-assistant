import os
from langchain_community.tools import DuckDuckGoSearchRun
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv('../infra/.env')

def optimize_query_for_search(user_query: str) -> str:
    """Utilise un LLM pour transformer une question conversationnelle en mots-clés SEO."""
    system_prompt = """
    Tu es un expert en recherche web (SEO). Ton unique but est de transformer la demande de l'utilisateur en une requête de recherche très courte, constituée uniquement de mots-clés pertinents.
    
    Règles :
    1. Retire tous les mots de politesse ou conversationnels ("propose moi", "je veux", "peux-tu").
    2. Ne réponds QUE par les mots-clés. Aucune phrase, aucune ponctuation superflue.
    3. Si la requête est déjà courte, laisse-la telle quelle.
    """
    
    optimizer = Agent(
        model=Groq(id="openai/gpt-oss-120b"), # Utilisation du même modèle que tes autres agents
        description=system_prompt
    )
    
    try:
        response = optimizer.run(user_query)
        # Nettoyage des éventuels guillemets renvoyés par le LLM
        optimized = response.content.strip().replace('"', '').replace("'", "")
        return optimized
    except Exception:
        return user_query # Fallback sur la question originale en cas d'erreur

def perform_web_search(query: str) -> str:
    """
    Outil de recherche Web avec reformulation de requête intégrée.
    """
    # 1. Optimisation de la requête
    optimized_query = optimize_query_for_search(query)
    print(f"🔍 Requête reformulée pour le Web : '{optimized_query}'...")
    
    try:
        # 2. Exécution de la recherche avec DuckDuckGo (plus efficace que Wikipedia pour des recommandations)
        search_tool = DuckDuckGoSearchRun()
        results = search_tool.invoke(optimized_query)
        
        if not results:
            return "Aucun résultat pertinent trouvé sur le Web."
            
        print("✅ Résultats Web récupérés avec succès !")
        return results
        
    except Exception as e:
        print(f"❌ Erreur lors de la recherche Web : {str(e)}")
        return "Impossible d'accéder à Internet pour le moment."

if __name__ == "__main__":
    # Test local rapide
    test_query = "propose moi des site web ou bien des livre pour etudier git"
    resultat = perform_web_search(test_query)
    
    print("\n--- Résultat de la recherche ---")
    print(resultat)