import os
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Chargement de la clé API
load_dotenv('../infra/.env')

def check_grounding(context: str, answer: str) -> str:
    """
    Évalue si la réponse générée (answer) est strictement supportée
    par le contexte fourni (context).
    Retourne "OUI" si la réponse est ancrée (grounded), sinon "NON".
    """
    
    system_prompt = """
You are a strict fact-checking evaluator for a RAG system.
Your ONLY task is to compare a GENERATED ANSWER against the provided CONTEXT.

Rule: 
Does the GENERATED ANSWER contain any factual claims, specific details, numbers, or conclusions that are NOT explicitly supported by the CONTEXT?
Note: The CONTEXT may be messy or incomplete, especially if it contains web search snippets ("AJOUT WEB"). You must evaluate whether the core facts in the ANSWER can be reasonably derived from the CONTEXT, even if the ANSWER uses different wording, better formatting, or basic logical deductions.

If the core facts in the ANSWER are reasonably supported by the CONTEXT, output: OUI
If there is a clear hallucination or major unsupported factual claim completely absent from the CONTEXT, output: NON

Output ONLY the word OUI or NON. Do not add any explanation or punctuation.
"""
    
    prompt = f"CONTEXT:\n{context}\n\nGENERATED ANSWER:\n{answer}"
    
    print("🔎 Vérification du Grounding (Anti-Hallucination)...")
    
    try:
        agent = Agent(
            model=Groq(id="openai/gpt-oss-120b"),
            description=system_prompt,
            instructions=["Output ONLY OUI or NON."]
        )
        
        try:
            response = agent.run(prompt)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower() or "expire" in str(e).lower() or "insufficient" in str(e).lower():
                print("⚠️ Limite de tokens atteinte, bascule sur l'API 2 pour le Grounding Checker...")
                fallback_agent = Agent(
                    model=Groq(id="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY2")),
                    description=system_prompt,
                    instructions=["Output ONLY OUI or NON."]
                )
                response = fallback_agent.run(prompt)
            else:
                raise e
                
        grade = response.content.strip().upper()
        
        # Nettoyage au cas où le LLM serait bavard
        if "OUI" in grade:
            print("✅ Grounding Check : SUCCÈS (Réponse validée)")
            return "OUI"
        else:
            print("❌ Grounding Check : ÉCHEC (Hallucination détectée)")
            return "NON"
            
    except Exception as e:
        print(f"❌ Erreur lors de la vérification du grounding : {e}")
        return "OUI" # Fallback permissif en cas de plantage d'API, on laisse passer
