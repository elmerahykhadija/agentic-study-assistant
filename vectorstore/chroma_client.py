import os
import shutil
import chromadb
from chromadb.utils import embedding_functions

# Le chemin pointe vers le volume monté dans ton docker-compose
CHROMA_PATH = "../chroma_db"

def get_chroma_collection():
    """
    Initialise la connexion à ChromaDB et configure le modèle d'embedding.
    """
    # Utilisation du client persistant pour que les données survivent au redémarrage de Docker
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Modèle d'embedding recommandé dans la roadmap (performant pour le multilingue FR/EN/AR)
    # Remplacé par paraphrase-multilingual-MiniLM-L12-v2 pour une vectorisation beaucoup plus rapide
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # On crée une "collection" (l'équivalent d'une table SQL) pour stocker les cours
    collection = client.get_or_create_collection(
        name="study_materials",
        embedding_function=embedding_func
    )
    
    return collection

def store_chunks_in_db(chunks, progress_callback=None):
    """
    Prend les chunks générés par le pipeline d'ingestion et les sauvegarde dans ChromaDB.
    """
    if not chunks:
        print("⚠️ Aucun chunk à stocker.")
        if progress_callback: progress_callback("Aucun document trouvé.", 1.0)
        return False
        
    collection = get_chroma_collection()
    
    # Préparation des listes requises par ChromaDB
    documents = [chunk['content'] for chunk in chunks]
    metadatas = [{"source": chunk['source']} for chunk in chunks]
    # Création d'un ID unique pour chaque chunk (ex: "mon_cours.pdf_0")
    ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in chunks]
    
    print(f"🧠 Vectorisation et insertion de {len(chunks)} chunks dans ChromaDB...")
    if progress_callback: progress_callback("Préparation de la vectorisation...", 0.6)
    
    # Insertion par lots (batching) pour éviter les surcharges de RAM (OOM) et montrer la progression
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        msg = f"🔄 Traitement du lot {i} à {end} sur {len(documents)}..."
        print(msg)
        if progress_callback:
            # Plage 0.6 -> 0.9 pour le stockage
            progress = 0.6 + (0.3 * (end / len(documents)))
            progress_callback(msg, progress)
            
        collection.upsert(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
    
    print("✅ Stockage vectoriel terminé avec succès !")
    if progress_callback: progress_callback("Vectorisation et stockage terminés !", 1.0)
    return True

def retrieve_context(query_text, n_results=4):
    """
    Recherche les chunks les plus pertinents par rapport à la question de l'utilisateur.
    """
    collection = get_chroma_collection()
    
    print(f"🔍 Recherche dans le cours pour la question : '{query_text}'")
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    # results['documents'][0] contient la liste des textes trouvés
    return results['documents'][0]

def clear_database():
    """
    Supprime physiquement le dossier ChromaDB et le dossier data/ pour repartir de zéro.
    """
    print("🧹 Purge de la base de données et des fichiers locaux...")
    
    # 1. Supprimer ChromaDB
    chroma_abs_path = os.path.abspath(CHROMA_PATH)
    if os.path.exists(chroma_abs_path):
        try:
            shutil.rmtree(chroma_abs_path)
            print(f"🗑️ Dossier supprimé : {chroma_abs_path}")
        except Exception as e:
            print(f"❌ Erreur lors de la suppression de ChromaDB : {e}")
            
    # 2. Vider le dossier data/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"❌ Erreur lors de la suppression de {file_path} : {e}")
        print(f"🗑️ Contenu du dossier vidé : {data_dir}")
        
    return True