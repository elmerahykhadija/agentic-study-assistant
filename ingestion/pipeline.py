import os
import gdown
import pymupdf  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def extract_text_with_ocr(pdf_path, use_ocr=False, progress_callback=None):
    """Extrait le texte via PyMuPDF et utilise Tesseract (OCR) pour les images ou scans."""
    doc = pymupdf.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Si la page contient très peu de texte, c'est probablement un scan ou une image
        if len(text.strip()) < 50:
            if use_ocr:
                msg = f"🔍 Scan détecté à la page {page_num + 1} de {os.path.basename(pdf_path)}. Lancement de l'OCR..."
                print(msg)
                if progress_callback: progress_callback(msg)
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Extraction du texte de l'image avec Tesseract
                    text += pytesseract.image_to_string(image, lang='fra+eng') + "\n"
            else:
                print(f"⏭️ Page {page_num + 1} ignorée (pas de texte et OCR désactivé).")
        
        full_text += f"\n--- Page {page_num + 1} ---\n" + text
        
    return full_text

def chunk_document(text, chunk_size=1000, chunk_overlap=200):
    """Découpe le texte avec LangChain pour préserver le contexte sémantique."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

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
        file = os.path.basename(file_path)
        print(f"📄 Traitement de : {file}")
        if progress_callback:
            # On réserve la plage 0.2 -> 0.6 pour le traitement des fichiers
            prog = 0.2 + (0.4 * (index / max(1, len(pdf_files))))
            progress_callback(f"Extraction texte : {file}", prog)
        
        # Extraction
        raw_text = extract_text_with_ocr(file_path, use_ocr, progress_callback)
        
        # Chunking
        chunks = chunk_document(raw_text)
        print(f"✂️ Document découpé en {len(chunks)} chunks.")
        
        # Ajout de métadonnées basiques
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": file,
                "chunk_id": i,
                "content": chunk
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