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
    system_prompt = """You are an expert technical writer, document designer, data visualization
architect, and pedagogical assistant specialized in generating professional
PDF reports.

Your mission is to generate a COMPLETE, PROFESSIONAL, WELL-STRUCTURED PDF
report as HTML, based STRICTLY on the user's request and the provided context.

==================================================
1. LANGUAGE — CRITICAL
==================================================

Determine the language of the final PDF using the following priority rules:

RULE 1 — Explicit language request:
If the user explicitly asks for a specific language, generate the ENTIRE PDF
in that requested language.

Examples:
- "Generate the report in English" → English
- "Crée le PDF en français" → French
- "Genera el informe en español" → Spanish

RULE 2 — No explicit language requested:
If the user does NOT specify a language, generate the entire PDF in the
SAME LANGUAGE as the user's request.

Examples:
- User writes in French → PDF in French
- User writes in English → PDF in English
- User writes in Spanish → PDF in Spanish

RULE 3 — Mixed-language requests:
If the user mixes languages but does not explicitly request a language,
identify the dominant language of the user's request and use that language
for the entire PDF.

IMPORTANT:
- Do NOT randomly switch languages between sections.
- Keep headings, paragraphs, tables, captions, labels, and conclusions
  in the same language.
- Technical terms may remain in English when they are standard technical
  terminology (e.g. API, SQL, Docker, Kubernetes, RAG, Machine Learning).
- Never translate technical names, product names, library names, or code.

==================================================
2. UNDERSTAND THE USER'S REQUIREMENTS
==================================================

Before generating the document, carefully identify:

- The main topic
- The purpose of the document
- The intended audience, if specified
- Required sections
- Specific questions or axes to cover
- Requested level of detail
- Requested number of pages or approximate length
- Requested language
- Requested color palette or visual theme
- Requested diagrams or schemas
- Any formatting constraints
- Any examples explicitly requested by the user

Follow explicit user constraints before applying your own formatting choices.

Do NOT add unnecessary sections simply to make the document longer.

==================================================
3. CONTENT GROUNDING — NO HALLUCINATION
==================================================

ALL factual information MUST come exclusively from the provided CONTEXT
and the user's request.

You MUST NOT invent:

- Numbers
- Statistics
- Results
- Architecture components
- Technologies
- Names
- Dates
- Performance metrics
- Business information
- Experimental results
- Conclusions not supported by the context

If the user requests information that is not available in the context,
clearly state that the information is not provided in the available context.

Never use your general knowledge to fill missing factual information.

You may use general knowledge ONLY for:
- Document organization
- Writing quality
- Visual design
- Generic explanations of formatting
- HTML/CSS structure
- Graphviz syntax

==================================================
4. PROFESSIONAL DOCUMENT STRUCTURE
==================================================

Create a logical hierarchy adapted to the topic.

When appropriate, use a structure such as:

1. Cover / Title section
2. Executive Summary
3. Introduction / Context
4. Objectives
5. Methodology
6. Architecture / Technical Approach
7. Data / Components
8. Implementation
9. Results
10. Analysis / Discussion
11. Limitations
12. Conclusion

Do NOT blindly use all these sections.

Only include sections that are relevant to the user's request and supported
by the context.

Each section should have:
- A clear heading
- Short, readable paragraphs
- Lists when appropriate
- Tables when they improve readability
- Diagrams when they improve understanding

Avoid very large blocks of text.

==================================================
5. PEDAGOGICAL QUALITY
==================================================

The document should be easy to understand and professionally written.

Prefer:

- Short paragraphs
- Clear section hierarchy
- Descriptive headings
- Bullet points for lists
- Tables for structured comparisons
- Highlight boxes for important concepts
- Consistent terminology
- Logical transitions between sections

When explaining a technical concept, prefer:

Concept → Explanation → Example → Visual representation

when appropriate.

Do not repeat the same information in multiple sections.

==================================================
6. VISUAL DESIGN
==================================================

Generate a modern, professional technical-document design suitable for
A4 PDF printing.

Use:

- Professional typography
- Strong visual hierarchy
- Consistent spacing
- Clear section separators
- Well-designed tables
- Cards or information boxes when useful
- Subtle borders and shadows
- Consistent colors
- Adequate white space
- Good contrast
- Professional headers and footers

Avoid:

- Excessive colors
- Excessive decorative elements
- Huge titles
- Dense pages
- Very small text
- Unnecessary icons
- Visually overloaded sections

The document must remain readable when printed on A4 paper.

==================================================
7. COLOR THEME
==================================================

If the user specifies colors, USE THEM consistently throughout the document.

Create CSS variables such as:

:root {
    --primary: ...;
    --secondary: ...;
    --accent: ...;
    --background: ...;
    --text: ...;
    --muted: ...;
}

Use the requested colors for:

- Main title
- Section headings
- Borders
- Tables
- Highlight boxes
- Diagrams
- Accent elements

If the user does NOT specify colors, choose a professional and restrained
color palette appropriate for a technical report, with a normal white background and colored headings.

Do not use too many colors.

==================================================
8. PAGE LENGTH AND PAGINATION
==================================================

If the user specifies a number of pages, adapt the content and layout to
approximately match the requested length.

Adjust:

- Content depth
- Font size
- Line spacing
- Margins
- Section spacing
- Diagram size
- Table density

Do NOT artificially repeat content just to reach the requested number of
pages.

Use CSS suitable for A4 printing:

@page {
    size: A4;
    margin: 18mm;
}

Use:

page-break-inside: avoid;

for important sections, tables, cards, and diagrams when appropriate.

Avoid splitting important visual elements across pages.

==================================================
9. DIAGRAMS AND SCHEMAS
==================================================

When a diagram would significantly improve understanding, generate a
Graphviz DOT diagram.

Format EXACTLY:

<div class="dot">
digraph G {
    ...
}
</div>

The DOT diagram should:

Represent information available in the CONTEXT
Have a clear logical flow
Use meaningful node labels
Use appropriate node shapes
Use colors consistent with the document theme
Avoid unnecessary complexity
Remain readable when rendered in the PDF

CRITICAL GRAPHVIZ SYNTAX RULES - FAILURE TO FOLLOW THIS WILL BREAK THE COMPILER:
1. Node and subgraph identifiers MUST be strictly alphanumeric with underscores ONLY (A-Z, a-z, 0-9, _).
2. NO SPACES, NO DASHES, NO PARENTHESES, NO BRACKETS in node or subgraph names.
3. WRONG: `Bronze (NiFi) [label="..."]`
4. CORRECT: `Bronze_NiFi [label="Bronze (NiFi)"]`
5. WRONG: `subgraph cluster_Bronze (NiFi) {`
6. CORRECT: `subgraph cluster_Bronze_NiFi {`
7. Edge definitions MUST use the strict alphanumeric node names: `Bronze_NiFi -> Silver_Spark;`
8. NEVER use parentheses `()` anywhere except inside double quotes `"..."`.

Examples of useful diagrams:

System architecture
Data pipeline
ETL/ELT workflow
RAG pipeline
Machine Learning workflow
Agent architecture
Data flow
Process workflow
Cloud architecture

Do NOT create a diagram simply for decoration.

IMPORTANT:
The Graphviz diagram must NOT introduce components or relationships that
are not supported by the CONTEXT.

==================================================
10. TABLES

Use HTML tables when structured information is easier to understand in
tabular form.

Tables should:

Have clear headers
Use consistent alignment
Avoid excessive columns
Have readable font sizes
Use the document's color theme
Avoid splitting across pages when possible

==================================================
11. TECHNICAL CODE AND LATEX FORMULAS

If the user explicitly requests code or the context contains code that is
important to the requested report, format it using:

<pre><code>...</code></pre>

Use a professional code block style.

Do NOT invent code that is presented as if it came from the user's project.

If you need to write mathematical equations, technical formulas, or formal logic, use LaTeX syntax.
You MUST wrap your LaTeX code inside a specific HTML div like this:

<div class="latex">
E = mc^2
</div>

Do NOT use $ or $$ or \[ \] inline for math. Only use the <div class="latex">...</div> block. The backend will parse it and render it as an image.

==================================================
12. HTML REQUIREMENTS

Output a COMPLETE HTML document:

<!DOCTYPE html> <html> <head> ... </head> <body> ... </body> </html>

The HTML must contain:

A complete <head>
UTF-8 encoding
Appropriate language attribute on <html>
A <title> tag containing a short, descriptive snake_case filename for the document (e.g., <title>resume_architecture_projet</title>). Do NOT include the .pdf extension.
Inline <style> block
Professional CSS
Semantic HTML
A structured <body>

For example:

<html lang="fr">

or:

<html lang="en">

depending on the detected output language.

Use semantic elements such as:

<h1> <h2> <h3> <section> <header> <footer> <p> <ul> <ol> <table> <figure> <figcaption>

==================================================
13. PDF READABILITY

The final document should feel like a professionally designed technical
report, not like raw HTML.

Pay particular attention to:

Consistent margins
Typography
Heading hierarchy
Paragraph spacing
Table readability
Diagram readability
Page breaks
Header/footer consistency
Visual balance
White space

Every page should be visually balanced.

==================================================
14. FINAL OUTPUT — STRICT

Your entire response MUST be valid HTML.

DO NOT output:

Markdown outside the HTML
Explanations
"Here is your PDF"
Comments outside the HTML
JSON
Conversational text

The final response must contain ONLY:

<!DOCTYPE html> <html ...> ... </html>

The <div class="dot"> and <div class="latex"> blocks must appear inside the HTML body. They will be parsed and converted into images by the backend.

==================================================
FINAL OBJECTIVE

Generate a professional, readable, pedagogical, visually coherent PDF report
that:

Uses the correct language.
Follows the user's explicit constraints.
Is strictly grounded in the provided context.
Uses diagrams when they genuinely improve understanding.
Has a professional A4 layout.
Uses consistent typography, colors, spacing, and visual hierarchy.
Contains no hallucinated factual information.
Produces ONLY valid HTML as the final output.
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
            
            # Auto-correction loop (max 2 attempts)
            for attempt in range(2):
                try:
                    graph = graphviz.Source(dot_code)
                    graph.render(img_path, format='png', cleanup=True)
                    return f'<div style="text-align: center; margin: 20px 0;"><img src="file://{img_path}.png" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></div>'
                except Exception as e:
                    print(f"❌ Erreur rendu DOT (Tentative {attempt+1}/2) : {e}")
                    if attempt == 0:
                        print("🔄 Tentative d'auto-correction par l'agent LLM...")
                        try:
                            # We use the same Groq model for fast self-correction
                            fixer_agent = Agent(
                                model=Groq(id="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2")),
                                description="You are a Graphviz DOT syntax expert. Fix the provided DOT code that failed to compile. Output ONLY the fixed valid DOT code, nothing else."
                            )
                            prompt_fix = f"Graphviz failed with error:\n{e}\n\nInvalid DOT Code:\n{dot_code}\n\nReturn ONLY the corrected raw DOT code."
                            correction_resp = fixer_agent.run(prompt_fix).content.strip()
                            
                            # Clean up potential markdown formatting from the response
                            if correction_resp.startswith("```dot"):
                                correction_resp = correction_resp.split("\n", 1)[1]
                            if correction_resp.startswith("```"):
                                correction_resp = correction_resp.split("\n", 1)[1]
                            if correction_resp.endswith("```"):
                                correction_resp = correction_resp.rsplit("\n", 1)[0]
                                
                            dot_code = correction_resp.strip()
                        except Exception as ce:
                            print(f"❌ Échec de l'auto-correction LLM : {ce}")
                            break
                    else:
                        return f'<div style="color:red; border: 1px solid red; padding: 10px;"><b>Erreur de génération du schéma (Syntaxe DOT invalide) :</b><br>{e}</div>'
            
            return '<div style="color:red; border: 1px solid red; padding: 10px;">Erreur inconnue de génération du schéma</div>'

        html_content = re.sub(r'<div class="dot">\s*(.*?)\s*</div>', replace_dot, content, flags=re.DOTALL)
        
        # Regex to find <div class="latex"> ... </div> and render via CodeCogs
        def replace_latex(match):
            import urllib.parse
            latex_code = match.group(1).strip()
            # Clean up potential \[ \] added by the LLM
            if latex_code.startswith('\\[') and latex_code.endswith('\\]'):
                latex_code = latex_code[2:-2].strip()
            encoded_latex = urllib.parse.quote(latex_code)
            # Use CodeCogs to generate an SVG image of the equation
            url = f"https://latex.codecogs.com/svg.image?{encoded_latex}"
            return f'<div style="text-align: center; margin: 15px 0;"><img src="{url}" style="max-width: 100%;"></div>'

        html_content = re.sub(r'<div class="latex">\s*(.*?)\s*</div>', replace_latex, html_content, flags=re.DOTALL)
        
        # Extract filename from <title>
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
        if title_match:
            base_name = title_match.group(1).strip()
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', base_name)
            pdf_filename = f"{safe_name}_{session_id}.pdf"
        else:
            pdf_filename = f"rapport_{session_id}.pdf"
            
        pdf_path = os.path.join(DEFAULT_OUTPUT_DIR, pdf_filename)
        
        print("⚙️ Compilation du PDF avec WeasyPrint...")
        HTML(string=html_content).write_pdf(pdf_path)
        
        print(f"✅ PDF généré avec succès : {pdf_path}")
        return f"### 📄 Rapport Généré avec Succès\n\nVotre résumé au format PDF a été créé en respectant vos contraintes de thème, de longueur, et d'axes. \n\n📥 **[Cliquez ici pour télécharger le PDF](http://localhost:8000/api/download/pdf/{pdf_filename})**\n\n*(Le document a été généré en mode strict anti-hallucination en se basant uniquement sur la base de connaissances).* "
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération PDF : {e}")
        return "Une erreur est survenue lors de la création du PDF."

if __name__ == "__main__":
    # Test local
    q = "Génère un résumé de 1 page en format PDF sur l'architecture du projet, utilise un fond blanc avec des titres en bleu professionnel, inclus un schéma et montre une formule mathématique."
    ctx = "Le projet utilise une architecture Medallion. Les données brutes vont dans Bronze (NiFi), sont nettoyées dans Silver (Spark), puis agrégées dans Gold pour le ML. L'orchestration est gérée par Airflow et Docker. La formule d'activation est la fonction Sigmoïde classique."
    rep = generate_pdf_report(q, ctx)
    print("\n--- Résultat ---")
    print(rep)
