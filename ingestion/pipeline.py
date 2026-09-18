import os
import gdown
import fitz  # PyMuPDF
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

def extract_text_with_ocr(pdf_path):
    """Extrait le texte via PyMuPDF et utilise Tesseract (OCR) pour les images ou scans."""
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Si la page contient très peu de texte, c'est probablement un scan ou une image
        if len(text.strip()) < 50:
            print(f"🔍 Scan détecté à la page {page_num + 1} de {os.path.basename(pdf_path)}. Lancement de l'OCR...")
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                
                # Extraction du texte de l'image avec Tesseract
                text += pytesseract.image_to_string(image, lang='fra+eng') + "\n"
        
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

def run_ingestion_pipeline(drive_url):
    """Orchestre tout le pipeline d'ingestion (Étape 1)."""
    print("🚀 Lancement du Pipeline d'Ingestion...\n")
    
    # 1. Collecte
    download_data(drive_url)
    
    # 2. Parcours des fichiers téléchargés
    all_chunks = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                print(f"📄 Traitement de : {file}")
                
                # Extraction
                raw_text = extract_text_with_ocr(file_path)
                
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