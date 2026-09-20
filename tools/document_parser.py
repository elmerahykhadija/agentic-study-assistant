import sys
import os
from PIL import Image
import pytesseract

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingestion.pipeline import extract_text_with_ocr, chunk_document
from vectorstore.chroma_client import store_chunks_in_db

def parse_local_document(file_path: str, session_id: str, use_ocr: bool = False, progress_callback = None) -> str:
    """Analyse un fichier local, le segmente et le stocke dans la base vectorielle."""
    try:
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        
        if progress_callback: progress_callback(f"Extraction du texte de {file_name}...", 0.2)
        
        raw_text = ""
        if ext == '.pdf':
            raw_text = extract_text_with_ocr(file_path, use_ocr=use_ocr, progress_callback=progress_callback)
        elif ext in ['.doc', '.docx']:
            if progress_callback: progress_callback(f"Extraction du texte du document Word...", 0.4)
            from unstructured.partition.auto import partition
            elements = partition(filename=file_path)
            raw_text = "\n\n".join([str(el) for el in elements])
        else:
            return f"Format de fichier non supporté: {ext}"
            
        if not raw_text.strip():
            return "Échec : Aucun texte n'a pu être extrait du fichier."

        if progress_callback: progress_callback("Découpage du document en segments...", 0.5)
        raw_chunks = chunk_document(raw_text)
        
        chunks = []
        for i, chunk in enumerate(raw_chunks):
            chunks.append({
                "source": file_name,
                "chunk_id": i,
                "content": chunk
            })
            
        if progress_callback: progress_callback("Enregistrement dans la base de données...", 0.7)
        success = store_chunks_in_db(chunks, session_id=session_id, progress_callback=progress_callback)
        
        if progress_callback: progress_callback("Terminé !", 1.0)
        
        if success:
            return f"Succès : Le fichier a été analysé et {len(chunks)} segments ont été mémorisés dans la base de données."
        else:
            return "Échec : Le fichier a été lu mais aucun segment n'a pu être créé."
            
    except Exception as e:
        return f"Erreur lors de l'analyse du fichier : {str(e)}"
