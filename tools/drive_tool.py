import sys
import os
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingestion.pipeline import run_ingestion_pipeline
from vectorstore.chroma_client import store_chunks_in_db # <-- NOUVEL IMPORT

def extract_course_from_drive(drive_url: str, use_ocr: bool = False, progress_callback = None) -> str:
    """Extrait le contenu d'un cours depuis Google Drive, le segmente et le stocke dans la base vectorielle."""
    try:
        # 1. Extraction et Chunking
        chunks = run_ingestion_pipeline(drive_url, use_ocr=use_ocr, progress_callback=progress_callback)
        
        # 2. Sauvegarde dans ChromaDB
        success = store_chunks_in_db(chunks, progress_callback=progress_callback)
        
        if success:
            return f"Succès : Le document a été lu et {len(chunks)} segments ont été mémorisés dans la base de données."
        else:
            return "Échec : Le document a été lu mais aucun segment n'a pu être créé."
            
    except Exception as e:
        return f"Erreur lors de l'extraction : {str(e)}"