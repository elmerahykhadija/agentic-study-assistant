import os
import gdown
import pymupdf  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# Configuration des dossiers
DATA_DIR = '../data/'

def download_data(url, output_dir=DATA_DIR):
    """Télécharge les fichiers ou dossiers depuis un lien Google Drive public."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"📥 Analyse du lien : {url}")
    
    if "drive.google.com/drive/folders/" in url or "folderview" in url:
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
    else:
        gdown.download(url, output=output_dir, quiet=False, fuzzy=True)
    print("✅ Téléchargement terminé.\n")

def extract_and_chunk_pymupdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Extrait le texte via PyMuPDF et le découpe (Stratégie rapide)."""
    doc = pymupdf.open(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks_with_meta = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            page_chunks = text_splitter.split_text(text)
            for chunk in page_chunks:
                chunks_with_meta.append({
                    "content": chunk,
                    "page": page_num + 1,
                    "heading": ""
                })
    return chunks_with_meta

def extract_and_chunk_unstructured(pdf_path, progress_callback=None):
    """Utilise unstructured (hi_res) pour extraire avec l'OCR et grouper par titre (Scans complexes)."""
    if progress_callback: progress_callback(f"Lancement d'Unstructured (OCR hi_res) sur {os.path.basename(pdf_path)}...", None)
    
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_bounding_boxes=True
        )
        
        chunks = chunk_by_title(
            elements,
            combine_text_under_n_chars=500,
            max_characters=1500,
        )
        
        chunks_with_meta = []
        for chunk in chunks:
            page_number = chunk.metadata.page_number if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'page_number') else 1
            chunks_with_meta.append({
                "content": str(chunk),
                "page": page_number,
                "heading": ""
            })
        return chunks_with_meta
    except Exception as e:
        print(f"⚠️ Erreur avec unstructured ({e}), bascule sur PyMuPDF...")
        return extract_and_chunk_pymupdf(pdf_path)

def run_ingestion_pipeline(drive_url, use_ocr=False, progress_callback=None):
    """Orchestre tout le pipeline d'ingestion (Étape 1)."""
    print("🚀 Lancement du Pipeline d'Ingestion...\n")
    if progress_callback: progress_callback("Initialisation de l'ingestion...", 0.05)
    
    # 1. Collecte
    if progress_callback: progress_callback("Téléchargement des données depuis Google Drive...", 0.1)
    download_data(drive_url)
    
    # 2. Parcours des fichiers téléchargés
    all_chunks = []
    pdf_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
                
    if progress_callback: progress_callback(f"Traitement de {len(pdf_files)} fichier(s) PDF...", 0.2)
                
    for index, file_path in enumerate(pdf_files):
        file_name = os.path.basename(file_path)
        print(f"📄 Traitement de : {file_name}")
        if progress_callback:
            # On réserve la plage 0.2 -> 0.6 pour le traitement des fichiers
            prog = 0.2 + (0.4 * (index / max(1, len(pdf_files))))
            progress_callback(f"Extraction texte : {file_name}", prog)
        
        # Extraction & Chunking
        if use_ocr:
            file_chunks = extract_and_chunk_unstructured(file_path, progress_callback)
        else:
            file_chunks = extract_and_chunk_pymupdf(file_path)
            
        print(f"✂️ Document découpé en {len(file_chunks)} chunks.")
        
        # Ajout de métadonnées enrichies
        for i, chunk in enumerate(file_chunks):
            all_chunks.append({
                "source": file_name,
                "chunk_id": i,
                "page": chunk.get("page", 1),
                "heading": chunk.get("heading", ""),
                "content": chunk["content"]
            })
            
    print(f"\n🎉 Ingestion terminée ! Total : {len(all_chunks)} segments de texte prêts.")
    if progress_callback: progress_callback(f"Ingestion terminée ({len(all_chunks)} segments).", 0.6)
    return all_chunks

if __name__ == '__main__':
    # Remplace par ton propre lien Drive public pour tester en local
    LIEN_TEST = "https://drive.google.com/drive/folders/14ijE-RU4_yBlAy-0nKTPXYBr-Q5WctFk" 
    
    if LIEN_TEST:
        chunks_prets = run_ingestion_pipeline(LIEN_TEST)
        
        # Afficher un aperçu du premier chunk pour vérifier
        if chunks_prets:
            print("\n👀 Aperçu du premier chunk :")
            print("-" * 40)
            print(chunks_prets[0]['content'])
            print("-" * 40)
    else:
        print("⚠️ N'oublie pas de mettre un vrai lien Drive dans la variable LIEN_TEST pour tester !")