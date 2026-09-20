import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingestion.pipeline import extract_and_chunk_pymupdf
from vectorstore.chroma_client import store_chunks_in_db
from langchain_text_splitters import RecursiveCharacterTextSplitter

def parse_local_document(file_path: str, session_id: str, progress_callback = None) -> str:
    """Analyse un fichier local, le segmente et le stocke dans la base vectorielle."""
    try:
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        
        if progress_callback: progress_callback(f"Extraction du texte de {file_name}...", 0.2)
        
        chunks = []
        if ext == '.pdf':
            file_chunks = extract_and_chunk_pymupdf(file_path)
                
            for i, chunk_data in enumerate(file_chunks):
                chunks.append({
                    "source": file_name,
                    "chunk_id": i,
                    "content": chunk_data["content"]
                })
        elif ext in ['.doc', '.docx']:
            if progress_callback: progress_callback(f"Extraction du texte du document Word...", 0.4)
            from unstructured.partition.auto import partition
            elements = partition(filename=file_path)
            raw_text = "\n\n".join([str(el) for el in elements])
            
            if not raw_text.strip():
                return "Échec : Aucun texte n'a pu être extrait du fichier."

            if progress_callback: progress_callback("Découpage du document en segments...", 0.5)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            raw_chunks = text_splitter.split_text(raw_text)
            
            for i, chunk in enumerate(raw_chunks):
                chunks.append({
                    "source": file_name,
                    "chunk_id": i,
                    "content": chunk
                })
        else:
            return f"Format de fichier non supporté: {ext}"
            
        if not chunks:
            return "Échec : Aucun texte n'a pu être extrait du fichier."
            
        if progress_callback: progress_callback("Enregistrement dans la base de données...", 0.7)
        success = store_chunks_in_db(chunks, session_id=session_id, progress_callback=progress_callback)
        
        if progress_callback: progress_callback("Terminé !", 1.0)
        
        if success:
            return f"Succès : Le fichier a été analysé et {len(chunks)} segments ont été mémorisés dans la base de données."
        else:
            return "Échec : Le fichier a été lu mais aucun segment n'a pu être créé."
            
    except Exception as e:
        return f"Erreur lors de l'analyse du fichier : {str(e)}"
