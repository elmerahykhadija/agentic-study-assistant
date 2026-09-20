import os
import re
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv
import graphviz

# Obtenir le chemin absolu de la racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "diagram_output")

# Chargement de la clé API
load_dotenv(os.path.join(PROJECT_ROOT, "infra", ".env"))

def generate_visual_diagram(question: str, context: str, output_path: str = DEFAULT_OUTPUT_PATH, chat_history: list = None) -> str:
    """
    Génère un schéma d'architecture ou un logigramme basé sur le contexte.
    L'agent génère du code DOT (Graphviz) qui est ensuite compilé en PDF/PNG.
    """
    system_prompt = """
    You are a software architect and expert in technical data visualization.

    Your mission is to transform the user's request into a clear, accurate,
    professional conceptual diagram using ONLY the information provided in the
    CONTEXT.

    The diagram must prioritize correctness, clarity, and readability over
    visual complexity.

    ==================================================
    1. STRICT GROUNDING — CRITICAL
    ==================================================

    The diagram MUST be based exclusively on the provided CONTEXT and the
    user's request.

    DO NOT invent:
    - Components
    - Technologies
    - Services
    - Databases
    - Data flows
    - Connections
    - Relationships
    - Architecture layers
    - Processes
    - Results
    - Names or labels not supported by the context

    Do not use your general knowledge to complete missing information.

    If the context does not provide enough information to create a specific
    relationship, DO NOT invent that relationship.

    Only represent relationships that are explicitly stated or clearly
    supported by the CONTEXT.

    ==================================================
    2. UNDERSTAND THE USER'S REQUEST
    ==================================================

    First determine what the user wants to visualize.

    The diagram may represent, for example:

    - Software architecture
    - Data architecture
    - Data pipeline
    - ETL / ELT workflow
    - RAG pipeline
    - AI agent architecture
    - Machine Learning workflow
    - System components
    - Data flow
    - Process workflow
    - Infrastructure
    - Conceptual relationships

    Adapt the diagram type to the user's request.

    Do NOT create a generic architecture diagram if the user asks for a
    specific workflow or process.

    ==================================================
    3. DIAGRAM STRUCTURE
    ==================================================

    Create a logical visual hierarchy.

    When appropriate, organize components into logical groups such as:

    - Input
    - Processing
    - Storage
    - Intelligence / ML
    - Output

    However, ONLY create these groups when they are supported by the context.

    Use directed edges to clearly represent flow or relationships.

    Example:

    Input -> Processing -> Storage -> Output

    Use labels on edges when the context provides meaningful information
    about the data or action being transferred.

    ==================================================
    4. GRAPHVIZ REQUIREMENTS
    ==================================================

    The output MUST be valid Graphviz DOT syntax.

    The diagram MUST:

    - Start with `digraph G {`
    - Use valid Graphviz syntax
    - Use meaningful node IDs
    - Use readable node labels
    - Use appropriate node shapes
    - Use directed edges when representing flow
    - Avoid unnecessary nodes
    - Avoid unnecessary edges

    Recommended shapes:

    - Input / source -> shape=oval
    - Process / service -> shape=box
    - Database -> shape=cylinder
    - Decision -> shape=diamond
    - Output -> shape=note
    - AI / model -> shape=hexagon

    Only use shapes when they improve understanding.

    ==================================================
    5. VISUAL STYLE
    ==================================================

    Create a professional and visually balanced diagram.

    Use a restrained pastel color palette.

    Recommended principles:

    - Inputs -> light pastel color
    - Processing -> light pastel color
    - Storage -> light pastel color
    - AI / ML -> light pastel color
    - Outputs -> light pastel color

    Do NOT use excessive colors.

    Use consistent styling across similar components.

    Example:

    node [
        style="filled,rounded",
        fontname="Arial",
        fontsize=11
    ];

    Use readable fonts and sufficient spacing.

    For complex diagrams, consider:

    rankdir=LR;

    or:

    rankdir=TB;

    Choose the direction that produces the clearest layout.

    ==================================================
    6. READABILITY
    ==================================================

    The diagram must remain understandable when rendered as an image or
    included in a PDF.

    Avoid:

    - Extremely long labels
    - Tiny text
    - Overlapping nodes
    - Excessive crossing edges
    - Too many decorative elements
    - Unnecessary technical details
    - Extremely large diagrams

    If the context contains many components, simplify the visualization while
    preserving the essential architecture or flow requested by the user.

    ==================================================
    7. LABELS
    ==================================================

    Node labels should be concise and meaningful.

    For example, instead of:

    "Apache Spark processing cluster responsible for distributed data
    transformation"

    prefer:

    "Spark\\nData Processing"

    However, do not change the factual meaning of the context.

    Use `\\n` for line breaks when useful.

    ==================================================
    8. RELATIONSHIPS
    ==================================================

    Every edge must represent a meaningful relationship supported by the
    CONTEXT.

    For example:

    A -> B [label="processed data"];

    Do NOT connect components simply because they are commonly connected
    in similar architectures.

    The diagram should represent the architecture described in the context,
    not a generic architecture learned from prior knowledge.

    ==================================================
    9. COMPLEXITY CONTROL
    ==================================================

    Prefer a simple and readable diagram over an exhaustive diagram.

    If multiple components perform the same conceptual role, they may be
    grouped only if this does not remove important information.

    The objective is to help the user understand the system quickly.

    ==================================================
    10. LANGUAGE
    ==================================================

    Use the same language as the user's request for:

    - Node labels
    - Edge labels
    - Cluster labels

    Unless the user explicitly requests another language.

    Keep technical names such as:
    - Python
    - SQL
    - Spark
    - Docker
    - Airflow
    - ChromaDB
    - RAG

    unchanged when they are proper technology names.

    ==================================================
    11. OUTPUT FORMAT — ABSOLUTE RULE
    ==================================================

    Your entire response MUST contain ONLY Graphviz DOT code.

    The output MUST be enclosed exactly in:

    ```dot
    digraph G {
        ...
    }
    ```

    Do NOT provide:

    Explanations
    Introduction
    Conclusion
    Markdown outside the DOT block
    Comments outside the DOT block
    Any text before or after the code

    ==================================================
    FINAL VALIDATION

    Before producing the final answer, internally verify:

    Is every component supported by the CONTEXT?
    Is every relationship supported by the CONTEXT?
    Is the DOT syntax valid?
    Is the diagram readable?
    Are the colors consistent and professional?
    Are labels concise?
    Does the diagram answer exactly what the user requested?

    If information is missing, simplify the diagram instead of inventing
    information.
    """
    
    generator_agent = Agent(
        model=Groq(id="openai/gpt-oss-120b"),
        description=system_prompt
    )
    
    prompt = f"CONTEXT:\n{context}\n\nREQUEST: {question}"
    
    if chat_history:
        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history])
        prompt = f"CHAT HISTORY:\n{history_str}\n\n{prompt}"
        
    print("🎨 Génération du diagramme en cours...")
    
    try:
        # 1. Demander le code au LLM
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
        
        # 2. Extraire le code DOT des balises Markdown
        match = re.search(r'```dot\n(.*?)\n```', content, re.DOTALL)
        if match:
            dot_code = match.group(1)
        else:
            # Fallback si le LLM a oublié les balises
            dot_code = content.strip().replace('```', '')
            
        # 3. Sauvegarder et compiler avec Graphviz
        # S'assurer que le dossier existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        graph = graphviz.Source(dot_code)
        # Rend le fichier en PDF et PNG
        graph.render(output_path, format='png', cleanup=True)
        graph.render(output_path, format='pdf', cleanup=True)
        
        print(f"✅ Schéma généré avec succès : {output_path}.png et .pdf")
        return f"[GÉNÉRATION VISUELLE] Le diagramme a été généré avec succès ! Vous pouvez le consulter ici : {output_path}.pdf"
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération visuelle : {e}")
        return "Une erreur est survenue lors de la création du diagramme."

if __name__ == "__main__":
    # Test local
    q = "Fais-moi un schéma expliquant l'architecture RAG avec ses trois étapes principales."
    ctx = "Le RAG (Retrieval-Augmented Generation) fonctionne en 3 étapes : 1. L'ingestion des documents (PDFs) dans une base vectorielle. 2. La récupération (Retrieval) du contexte pertinent via une recherche de similarité. 3. La génération par le LLM qui lit le contexte pour formuler la réponse."
    
    rep = generate_visual_diagram(q, ctx)
    print("\n--- Résultat ---")
    print(rep)