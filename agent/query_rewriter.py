import os
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv('../infra/.env')

def rewrite_query(original_query: str, chat_history: list = None) -> str:
    """
    Reformule la question de l'utilisateur pour optimiser la recherche
    vectorielle et lexicale (CRAG).
    """
    
    system_prompt = """
You are an expert query optimizer for a RAG (Retrieval-Augmented Generation) system.
Your task is to rewrite the user's QUESTION to make it more precise and optimized for both semantic and keyword-based search.

Guidelines:
1. Identify the core concepts and keywords in the question.
2. Remove conversational filler words ("peux-tu", "dis-moi", "bonjour").
3. Add relevant synonyms or technical terms if it clarifies the intent.
4. Keep the rewritten query concise and directly searchable.
5. Do NOT answer the question. ONLY output the reformulated query.

Example 1:
User: "Salut, c'est quoi le gradient descent en gros ?"
Output: "définition gradient descent algorithme optimisation"

Example 2:
User: "Comment on fait pour éviter l'overfitting ?"
Output: "techniques éviter overfitting surapprentissage méthodes régularisation"

OUTPUT FORMAT:
Return ONLY the reformulated query text. No quotes, no explanations.
"""
    
    # Construire un contexte optionnel avec l'historique
    history_context = ""
    if chat_history and len(chat_history) > 0:
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-3:]])
        history_context = f"\n\nRECENT CHAT HISTORY:\n{history_str}"
        
    prompt = f"QUESTION: {original_query}{history_context}"
    
    print(f"🔄 CRAG: Reformulation de la question '{original_query}'...")
    
    try:
        agent = Agent(
            model=Groq(id="openai/gpt-oss-120b"),
            description=system_prompt,
            instructions=["Output ONLY the reformulated query text."]
        )
        
        try:
            response = agent.run(prompt)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower() or "expire" in str(e).lower() or "insufficient" in str(e).lower():
                print("⚠️ Limite de tokens atteinte, bascule sur l'API 2 pour le rewriter...")
                fallback_agent = Agent(
                    model=Groq(id="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY2")),
                    description=system_prompt,
                    instructions=["Output ONLY the reformulated query text."]
                )
                response = fallback_agent.run(prompt)
            else:
                raise e
                
        rewritten = response.content.strip()
        print(f"✅ Nouvelle requête : '{rewritten}'")
        return rewritten
        
    except Exception as e:
        print(f"❌ Erreur lors de la reformulation : {e}")
        return original_query # Fallback sécurisé
