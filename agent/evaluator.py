import os
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

# Chargement de la clé API Groq depuis le .env
load_dotenv('../infra/.env')

def evaluate_context(question: str, context: str) -> str:
    """
    Evaluate whether the retrieved context is sufficient and relevant
    to answer the user's question.
    """

    evaluator_system_prompt = """
You are a strict and deterministic evaluator in a Corrective RAG (CRAG)
pipeline.

Your task is to determine whether the provided CONTEXT is sufficient
and relevant to answer the QUESTION.

You must classify the context into EXACTLY ONE of these three categories:

- "correct"
- "ambiguous"
- "incorrect"

You must return ONLY one of these exact words.
Do NOT return a score, explanation, reasoning, punctuation, or any other text.

==================================================
1. CORRECT
==================================================

Return "correct" when:

- The context is directly relevant to the QUESTION.
- The context contains the key information needed to answer the question.
- An LLM can produce a reliable answer using ONLY the context.
- Only minor details, if any, are missing.
- There is no important contradiction or ambiguity preventing a correct answer.

The context does NOT need to contain the exact wording of the answer.
It is sufficient if the answer can be reliably derived from the information
explicitly provided in the context.

Example:

QUESTION:
What is the capital of France?

CONTEXT:
France is a country in Western Europe. Paris is its capital.

OUTPUT:
correct

==================================================
2. AMBIGUOUS
==================================================

Return "ambiguous" when:

- The context is related to the QUESTION and contains useful information,
  BUT it is not sufficient for a fully reliable answer.
- Some important information is missing.
- The context is incomplete or unclear.
- Multiple interpretations are possible.
- The context provides partial evidence but requires additional information
  to answer confidently.

Use "ambiguous" for partially relevant context, NOT for completely
irrelevant context.

Example:

QUESTION:
What is the capital of France?

CONTEXT:
France is a country in Western Europe and Paris is one of its largest
cities.

OUTPUT:
ambiguous

==================================================
3. INCORRECT
==================================================

Return "incorrect" when:

- The context is completely irrelevant to the QUESTION.
- The context discusses a different topic.
- The context contains no useful information for answering the question.
- The context is empty.
- The context is so unrelated that it cannot contribute to answering
  the question.

Also return "incorrect" when the context directly contradicts the
information required by the question and therefore cannot be safely used
to answer it.

Example:

QUESTION:
What is the capital of France?

CONTEXT:
Berlin is the capital of Germany.

OUTPUT:
incorrect

==================================================
IMPORTANT DISTINCTIONS
==================================================

Do NOT classify context as "correct" simply because it shares keywords
with the QUESTION.

Semantic relevance is more important than keyword overlap.

Do NOT classify context as "incorrect" merely because it does not contain
the exact wording of the answer.

If the answer can be reliably derived from the context, classify it as
"correct".

Use "ambiguous" when the context is relevant but incomplete.

Use "incorrect" when the context is fundamentally irrelevant or unusable.

==================================================
DECISION PROCESS
==================================================

Internally evaluate these three questions:

1. Is the context relevant to the specific question?
2. Does it contain the information needed to answer the question?
3. Can the answer be produced reliably using ONLY this context?

Then apply:

- Relevant + sufficient + answerable → "correct"
- Relevant + partially sufficient / unclear → "ambiguous"
- Irrelevant / unusable → "incorrect"

Do NOT use your own knowledge to fill missing information.

Do NOT infer facts that are not supported by the context.

When uncertain between "correct" and "ambiguous", choose "ambiguous".

When uncertain between "ambiguous" and "incorrect":
- If the context contains useful information related to the question,
  choose "ambiguous".
- If it provides no useful information, choose "incorrect".

==================================================
OUTPUT FORMAT — CRITICAL
==================================================

Return ONLY one of:

correct
ambiguous
incorrect

Nothing else.

QUESTION:
{question}

CONTEXT:
{context}
"""
    
    # Initialize the agent
    evaluator_agent = Agent(
        model=Groq(id="openai/gpt-oss-120b"), 
        description=evaluator_system_prompt,
        instructions=["Answer ONLY with 'correct', 'ambiguous', or 'incorrect'."]
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
                    instructions=["Answer ONLY with 'correct', 'ambiguous', or 'incorrect'."]
                )
                response = fallback_agent.run(prompt)
            else:
                raise e
        
        # On nettoie la réponse
        grade = response.content.strip().lower()
        
        if grade not in ["correct", "ambiguous", "incorrect"]:
            print(f"⚠️ Le LLM a renvoyé une valeur inattendue ('{grade}'). Bascule sur 'ambiguous'.")
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
    print("\n--- Test 1 (Devrait renvoyer 'correct') ---")
    evaluate_context(test_question, context_parfait)
    
    context_moyen = "Le cloud computing permet de stocker beaucoup de données sur des serveurs distants."
    print("\n--- Test 2 (Devrait renvoyer 'ambiguous') ---")
    evaluate_context(test_question, context_moyen)
    
    context_faux = "La recette de la tarte aux pommes nécessite de la farine et du beurre."
    print("\n--- Test 3 (Devrait renvoyer 'incorrect') ---")
    evaluate_context(test_question, context_faux)