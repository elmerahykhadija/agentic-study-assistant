import os
import re
import uuid
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv
import graphviz
from weasyprint import HTML

# Obtenir le chemin absolu de la racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "pdf_output")

# Chargement de la clé API
load_dotenv(os.path.join(PROJECT_ROOT, "infra", ".env"))

def generate_pdf_report(question: str, context: str, chat_history: list = None) -> str:
    """
    Génère un rapport PDF structuré contenant du texte et des schémas.
    L'agent extrait les contraintes dynamiquement depuis le prompt utilisateur.
    """
    system_prompt = """You are an expert technical writer, data visualization architect, and strict pedagogical assistant.
Your mission is to generate a comprehensive, visually appealing PDF report (in HTML format) based STRICTLY on the user's request and the provided context.

CRITICAL INSTRUCTIONS:
1. Extract User Constraints: Read the user's request carefully. Identify and apply:
   - Any specific color theme requested. Generate CSS variables reflecting these colors.
   - Specific focus axes (e.g., context, architecture, results, analysis). Structure your headings based on these.
   - An estimated length (e.g., number of pages). Adapt your content depth, font sizes, and spacing to match the requested length.
   
2. HTML Structure: Output a complete HTML document (from <!DOCTYPE html><html> to </html>). 
   - Use semantic tags (<h1>, <section>, <ul>, <p>). 
   - Include an inline <style> block with professional, modern CSS (flexbox, shadows, modern typography, margins for A4 pages).
   - Ensure the styling supports printing (e.g., page-break-inside: avoid for sections).

3. Strict Anti-Hallucination (Grounding): 
   - You MUST base ALL your facts, numbers, architectural details, and analysis EXCLUSIVELY on the provided CONTEXT. 
   - Do NOT invent information. If the context does not contain enough information for a requested axis, state it clearly.

4. Mixed Media (Diagrams): The user wants both text and schemas. 
   - When a section would benefit from a visual representation, you MUST include Graphviz DOT code inside a markdown code block.
   - Format: ```dot\n digraph G { ... } \n```
   - Ensure the DOT code uses node shapes and colors that match the requested theme.
   - Do not place the dot block inside HTML attributes. Just place it inline in the HTML body where the image should appear.

5. Output Format: Do NOT output any conversational text outside the HTML tags. Your entire response must be valid HTML. The dot blocks will be parsed and replaced with images by the backend.
"""
    
    generator_agent = Agent(
        model=Groq(id="openai/gpt-oss-120b"),
        description=system_prompt
    )
    
    prompt = f"CONTEXT:\n{context}\n\nUSER REQUEST: {question}"
    
    if chat_history:
        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history])
        prompt = f"CHAT HISTORY:\n{history_str}\n\n{prompt}"
        
    print("📄 Génération du rapport PDF en cours...")
    
    try:
        try:
            response = generator_agent.run(prompt)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower() or "expire" in str(e).lower() or "insufficient" in str(e).lower():
                print("⚠️ Limite de tokens atteinte, bascule sur l'API 2...")
                fallback_agent = Agent(
                    model=Groq(id="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY2")),
                    description=system_prompt
                )
                response = fallback_agent.run(prompt)
            else:
                raise e
        content = response.content
        
        # Extract HTML block if the model wraps it in markdown ```html
        html_match = re.search(r'```html\n(.*?)\n```', content, re.DOTALL)
        if html_match:
            content = html_match.group(1)
            
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        session_id = str(uuid.uuid4())[:8]
        
        # Regex to find ```dot ... ```
        def replace_dot(match):
            dot_code = match.group(1).strip()
            img_filename = f"schema_{uuid.uuid4().hex[:6]}"
            img_path = os.path.join(DEFAULT_OUTPUT_DIR, img_filename)
            
            try:
                graph = graphviz.Source(dot_code)
                graph.render(img_path, format='png', cleanup=True)
                return f'<div style="text-align: center; margin: 20px 0;"><img src="file://{img_path}.png" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></div>'
            except Exception as e:
                print(f"❌ Erreur rendu DOT : {e}")
                return '<div style="color:red; border: 1px solid red; padding: 10px;">Erreur de génération du schéma</div>'

        html_content = re.sub(r'```dot\s*(.*?)\s*```', replace_dot, content, flags=re.DOTALL)
        
        pdf_filename = f"rapport_{session_id}.pdf"
        pdf_path = os.path.join(DEFAULT_OUTPUT_DIR, pdf_filename)
        
        print("⚙️ Compilation du PDF avec WeasyPrint...")
        HTML(string=html_content).write_pdf(pdf_path)
        
        print(f"✅ PDF généré avec succès : {pdf_path}")
        return f"### 📄 Rapport Généré avec Succès\n\nVotre résumé au format PDF a été créé en respectant vos contraintes de thème, de longueur, et d'axes. \n\n📥 **[Cliquez ici pour consulter le rapport PDF](file://{pdf_path})**\n\n*(Le document a été généré en mode strict anti-hallucination en se basant uniquement sur la base de connaissances).* "
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération PDF : {e}")
        return "Une erreur est survenue lors de la création du PDF."

if __name__ == "__main__":
    # Test local
    q = "Génère un résumé de 1 page en format PDF sur l'architecture du projet, utilise un thème sombre avec du vert pastel, et inclus un schéma."
    ctx = "Le projet utilise une architecture Medallion. Les données brutes vont dans Bronze (NiFi), sont nettoyées dans Silver (Spark), puis agrégées dans Gold pour le ML. L'orchestration est gérée par Airflow et Docker."
    rep = generate_pdf_report(q, ctx)
    print("\n--- Résultat ---")
    print(rep)
