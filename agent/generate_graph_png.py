import sys
import os

# Ajout du chemin racine
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from agent.graph import app

output_path = os.path.join(PROJECT_ROOT, "imgs", "graph_architecture2.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

try:
    png_data = app.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_data)
    print(f"Graph PNG generated successfully at {output_path}")
except Exception as e:
    print(f"Failed to generate PNG: {e}")
