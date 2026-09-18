import os
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
    # Note : Lors du premier lancement, le modèle sera téléchargé automatiquement.
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3"
    )
    
    # On crée une "collection" (l'équivalent d'une table SQL) pour stocker les cours
    collection = client.get_or_create_collection(
        name="study_materials",
        embedding_function=embedding_func
    )
    
    return collection

def store_chunks_in_db(chunks):
    """
    Prend les chunks générés par le pipeline d'ingestion et les sauvegarde dans ChromaDB.
    """
    if not chunks:
        print("⚠️ Aucun chunk à stocker.")
        return False
        
    collection = get_chroma_collection()
    
    # Préparation des listes requises par ChromaDB
    documents = [chunk['content'] for chunk in chunks]
    metadatas = [{"source": chunk['source']} for chunk in chunks]
    # Création d'un ID unique pour chaque chunk (ex: "mon_cours.pdf_0")
    ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in chunks]
    
    print(f"🧠 Vectorisation et insertion de {len(chunks)} chunks dans ChromaDB...")
    
    # Insertion dans la base
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print("✅ Stockage vectoriel terminé avec succès !")
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