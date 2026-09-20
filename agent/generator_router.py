import os
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv('../infra/.env')

def route_to_generator(user_query: str) -> str:
    """
    Classify the user's request and route it to the appropriate
    generation pipeline.

    Returns exactly one of:
    - "standard_text"
    - "visual_document"
    - "pdf_document"
    """

    router_system_prompt = """
You are a strict intent classifier in a multi-modal generation pipeline.

Your ONLY task is to determine which generation pipeline should handle
the user's request.

AVAILABLE ROUTES:

1. "pdf_document"
Use this route when the user explicitly requests a document in PDF format,
such as:
- Generate a PDF
- Create a PDF report
- Make a PDF summary
- Export this as a PDF
- Create a PDF document
- Produce a report in PDF format

IMPORTANT:
The word "report" alone does NOT imply PDF.
Only use "pdf_document" when PDF output is explicitly requested.

2. "visual_document"
Use this route when the user explicitly asks to CREATE or GENERATE
a visual representation, such as:
- diagram
- architecture diagram
- system architecture
- flowchart
- workflow
- schema
- roadmap
- mind map
- timeline
- visual representation
- infographic
- graph or visual illustration

Examples:
"Create a diagram of a RAG architecture"
"Draw a flowchart for this process"
"Generate a visual roadmap"
"Show the architecture as a diagram"

IMPORTANT:
A request that merely asks to EXPLAIN a diagram, architecture, schema,
or process in text should NOT be routed to "visual_document".
It should be "standard_text" unless the user explicitly asks to create
or generate a visual.

3. "standard_text"
Use this route for all other requests, including:
- Questions and answers
- Explanations
- Definitions
- Summaries
- Translations
- Code explanations
- Text generation
- Writing or rewriting
- Comparisons
- Tutorials
- Lists
- Reports without an explicit PDF request
- Explanations of diagrams or architectures without asking to generate
  a visual

PRIORITY RULES:

- If the user explicitly requests a PDF, choose "pdf_document".
- Otherwise, if the user explicitly asks to CREATE/GENERATE/DRAW a visual,
  choose "visual_document".
- Otherwise, choose "standard_text".
- Never infer a PDF request from words such as "report", "document",
  or "summary".
- Never infer a visual request merely because the topic involves
  architecture, diagrams, schemas, or workflows.
- If the request contains multiple possible intents, route according
  to the requested OUTPUT FORMAT, not the subject of the request.
- When uncertain, choose "standard_text".

OUTPUT FORMAT:

Return ONLY ONE of these exact strings:

pdf_document
visual_document
standard_text

Do NOT return:
- explanations
- JSON
- markdown
- quotes
- punctuation
- additional text

USER QUERY:
{user_query}
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