from langchain_community.tools import DuckDuckGoSearchRun

def perform_web_search(query: str) -> str:
    """
    Outil de recherche Web. À déclencher UNIQUEMENT lorsque l'évaluateur CRAG
    juge que le contexte issu des documents est 'incorrect' ou 'ambiguous'.
    
    Args:
        query (str): La question de l'utilisateur reformulée pour une recherche web.
        
    Returns:
        str: Les extraits pertinents trouvés sur Internet.
    """
    print(f"🌐 Lancement de la recherche Web de secours pour : '{query}'...")
    
    try:
        # Initialisation de l'outil DuckDuckGo de LangChain
        search_tool = DuckDuckGoSearchRun()
        
        # Exécution de la recherche
        results = search_tool.invoke(query)
        
        if not results:
            return "Aucun résultat pertinent trouvé sur le Web."
            
        print("✅ Résultats Web récupérés avec succès !")
        return results
        
    except Exception as e:
        print(f"❌ Erreur lors de la recherche Web : {str(e)}")
        return "Impossible d'accéder à Internet pour le moment."

if __name__ == "__main__":
    # Test local rapide
    test_query = "Quelle est la date de sortie du modèle Llama 3 de Meta ?"
    resultat = perform_web_search(test_query)
    
    print("\n--- Résultat de la recherche ---")
    print(resultat)