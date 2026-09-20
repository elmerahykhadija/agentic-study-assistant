import os
import concurrent.futures
from agno.agent import Agent
from agno.models.groq import Groq
import re
from dotenv import load_dotenv
import sys

# Chargement du .env pour les clés d'API
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, 'infra', '.env')
load_dotenv(env_path)

class LLMReranker:
    """Reranker utilisant un LLM (Groq) comme juge pour évaluer la pertinence."""
    def __init__(self, model_id="openai/gpt-oss-120b"):
        self.model_id = model_id
        
        self.system_prompt = """
You are an expert relevance ranker. Given a user QUESTION and a DOCUMENT chunk, your task is to score the relevance of the DOCUMENT to the QUESTION on a scale from 0 to 100.
Score 100: The document explicitly and fully answers the question.
Score 50: The document is somewhat related but doesn't fully answer it.
Score 0: The document is completely irrelevant.
Output ONLY a single integer between 0 and 100. Do not output anything else.
"""

    def _score_doc(self, query, doc):
        prompt = f"QUESTION: {query}\n\nDOCUMENT:\n{doc['content']}"
        score = 0
        try:
            agent = Agent(
                model=Groq(id=self.model_id),
                description=self.system_prompt,
                instructions=["Output ONLY an integer between 0 and 100."]
            )
            try:
                response = agent.run(prompt)
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower() or "expire" in str(e).lower() or "insufficient" in str(e).lower():
                    fallback_agent = Agent(
                        model=Groq(id=self.model_id, api_key=os.getenv("GROQ_API_KEY2")),
                        description=self.system_prompt,
                        instructions=["Output ONLY an integer between 0 and 100."]
                    )
                    response = fallback_agent.run(prompt)
                else:
                    raise e
                    
            # Parse score
            score_str = response.content.strip()
            digits = re.findall(r'\d+', score_str)
            if digits:
                score = int(digits[0])
                score = min(max(score, 0), 100)
        except Exception as e:
            print(f"❌ Erreur lors du LLM reranking: {e}")
            
        doc["rerank_score"] = float(score)
        return doc

    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []
            
        print(f"⚖️ LLM Reranker (Groq) évalue {len(documents)} documents en parallèle...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self._score_doc, query, doc) for doc in documents]
            concurrent.futures.wait(futures)
            
        # Trier par score descendant
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        return documents[:top_k]

# Instance globale pour un import direct
reranker = LLMReranker()
