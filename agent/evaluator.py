import os
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Chargement de la clé API Groq depuis le .env
load_dotenv('../infra/.env')

def evaluate_context(question: str, context: str) -> str:
    """
    Évalue si le contexte extrait de ChromaDB est suffisant pour répondre à la question.
    Utilise un score de pertinence :
    - Score >= 0.7 : 'correct'
    - Score < 0.3 : 'incorrect'
    - Entre les deux : 'ambiguous'
    """
    # New prompt: We ask for a strict numerical score
    evaluator_system_prompt = """
    You are a strict evaluator for a Corrective RAG (CRAG) architecture. 
    Your mission is to evaluate the relevance of the provided CONTEXT with respect to the user's QUESTION.
    
    Evaluation Rules:
    You must assign a relevance score as a decimal number between 0.0 and 1.0.
    - 1.0 means the context contains the exact and complete answer.
    - 0.0 means the context is completely irrelevant.
    
    CRITICAL: You must answer ONLY with the number (e.g., 0.8, 0.2, 0.5). Do not add any text or explanation.
    """
    
    # Initialize the agent
    evaluator_agent = Agent(
        model=Groq(id="openai/gpt-oss-120b"), 
        description=evaluator_system_prompt,
        instructions=["Answer only with a decimal number (e.g., 0.75)."]
    )
    
    prompt = f"QUESTION: {question}\n\nCONTEXT: {context}"
    
    print(f"⚖️ CRAG Evaluator analyse le contexte...")
    
    try:
        try:
            response = evaluator_agent.run(prompt)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower() or "expire" in str(e).lower() or "insufficient" in str(e).lower():
                print("⚠️ Limite de tokens atteinte, bascule sur l'API 2...")
                fallback_agent = Agent(
                    model=Groq(id="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY2")), 
                    description=evaluator_system_prompt,
                    instructions=["Answer only with a decimal number (e.g., 0.75)."]
                )
                response = fallback_agent.run(prompt)
            else:
                raise e
        
        # On nettoie la réponse pour ne garder que le chiffre
        score_str = response.content.strip()
        
        try:
            # Conversion en nombre décimal
            score = float(score_str)
        except ValueError:
            print(f"⚠️ Le LLM n'a pas renvoyé un nombre valide ('{score_str}'). Bascule sur 'ambiguous'.")
            return "ambiguous"
            
        print(f"📊 Score de pertinence calculé : {score}")
        
        # --- TA LOGIQUE DE SEUILS ICI ---
        if score >= 0.7:
            grade = "correct"
        elif score < 0.3:
            grade = "incorrect"
        else:
            grade = "ambiguous"
            
        print(f"✅ Résultat de l'évaluation : {grade.upper()}")
        return grade
        
    except Exception as e:
        print(f"❌ Erreur lors de l'évaluation Groq : {e}")
        return "ambiguous" # Fallback sécurisé

if __name__ == "__main__":
    # Tests locaux
    test_question = "Qu'est-ce que le Big Data ?"
    
    context_parfait = "Le Big Data désigne les ensembles de données devenus si volumineux qu'ils dépassent les capacités de capture, de stockage et d'analyse des outils classiques."
    print("\n--- Test 1 (Devrait avoir un score >= 0.7) ---")
    evaluate_context(test_question, context_parfait)
    
    context_moyen = "Le cloud computing permet de stocker beaucoup de données sur des serveurs distants."
    print("\n--- Test 2 (Devrait être ambigu, entre 0.3 et 0.7) ---")
    evaluate_context(test_question, context_moyen)
    
    context_faux = "La recette de la tarte aux pommes nécessite de la farine et du beurre."
    print("\n--- Test 3 (Devrait avoir un score < 0.3) ---")
    evaluate_context(test_question, context_faux)