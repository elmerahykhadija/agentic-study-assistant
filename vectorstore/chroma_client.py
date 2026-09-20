import os
import shutil
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import sys
import os

# Ajout du dossier racine au PATH pour l'import de agent.reranker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.reranker import reranker

# Configuration parameters (can be adjusted or moved to .env)
VECTOR_K = 15
BM25_K = 15
RERANK_K = 5
RERANKER_MODEL = "openai/gpt-oss-120b" # LLM Reranker via Groq

# Chargement du .env pour les clés d'API (au cas où)
from dotenv import load_dotenv
load_dotenv('../infra/.env')

# Le chemin pointe vers le volume monté dans ton docker-compose
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    collection = client.get_or_create_collection(
        name="study_materials",
        embedding_function=embedding_func
    )
    return collection

class BM25Index:
    """Gère un index BM25 en mémoire, synchronisé avec ChromaDB au démarrage."""
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.metadatas = []
        self.ids = []
        self._load_from_chroma()

    def _load_from_chroma(self):
        collection = get_chroma_collection()
        all_data = collection.get() # fetch all documents
        if all_data and all_data['documents']:
            self.documents = all_data['documents']
            self.metadatas = all_data['metadatas']
            self.ids = all_data['ids']
            self._rebuild_index()

    def _rebuild_index(self):
        if self.documents:
            tokenized_docs = [doc.lower().split(" ") for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None

    def add_documents(self, docs, metas, doc_ids):
        self.documents.extend(docs)
        self.metadatas.extend(metas)
        self.ids.extend(doc_ids)
        self._rebuild_index()

    def search(self, query, session_id, top_k=BM25_K):
        if not self.bm25:
            return []
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        # Associer les scores aux documents
        results = []
        for i, score in enumerate(scores):
            # Filtrer par session_id
            if self.metadatas[i].get("session_id") == session_id:
                results.append({
                    "id": self.ids[i],
                    "content": self.documents[i],
                    "metadata": self.metadatas[i],
                    "score": score
                })
        
        # Trier par score BM25 descendant
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# Instance globale pour le BM25 en mémoire
bm25_index = BM25Index()



def store_chunks_in_db(chunks, session_id: str, progress_callback=None):
    if not chunks:
        print("⚠️ Aucun chunk à stocker.")
        if progress_callback: progress_callback("Aucun document trouvé.", 1.0)
        return False
        
    collection = get_chroma_collection()
    
    documents = [chunk['content'] for chunk in chunks]
    metadatas = [{
        "source": chunk['source'], 
        "session_id": session_id, 
        "page": chunk.get('page', 1),
        "heading": chunk.get('heading', "")
    } for chunk in chunks]
    ids = [f"{session_id}_{chunk['source']}_{chunk['chunk_id']}" for chunk in chunks]
    
    print(f"🧠 Vectorisation et insertion de {len(chunks)} chunks dans ChromaDB et BM25...")
    if progress_callback: progress_callback("Préparation de la vectorisation...", 0.6)
    
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        msg = f"🔄 Traitement du lot {i} à {end} sur {len(documents)}..."
        print(msg)
        if progress_callback:
            progress = 0.6 + (0.3 * (end / len(documents)))
            progress_callback(msg, progress)
            
        collection.upsert(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        
    # Mettre à jour l'index BM25
    bm25_index.add_documents(documents, metadatas, ids)
    
    print("✅ Stockage vectoriel et lexical terminé avec succès !")
    if progress_callback: progress_callback("Vectorisation et stockage terminés !", 1.0)
    return True

def retrieve_context(query_text, session_id: str, return_dicts=False):
    """
    Recherche Hybride : ChromaDB + BM25, puis Reranking.
    """
    print(f"🔍 Recherche hybride pour : '{query_text}'")
    
    # 1. Recherche Vectorielle (ChromaDB)
    collection = get_chroma_collection()
    vector_results = collection.query(
        query_texts=[query_text],
        n_results=VECTOR_K,
        where={"session_id": session_id}
    )
    
    combined_docs = {} # id -> doc_dict
    
    if vector_results and vector_results['documents'] and vector_results['documents'][0]:
        for i in range(len(vector_results['documents'][0])):
            doc_id = vector_results['ids'][0][i]
            combined_docs[doc_id] = {
                "id": doc_id,
                "content": vector_results['documents'][0][i],
                "metadata": vector_results['metadatas'][0][i] if vector_results['metadatas'] else {}
            }
            
    # 2. Recherche Lexicale (BM25)
    bm25_results = bm25_index.search(query_text, session_id, top_k=BM25_K)
    for res in bm25_results:
        if res["id"] not in combined_docs:
            combined_docs[res["id"]] = res
            
    docs_to_rerank = list(combined_docs.values())
    print(f"🔄 Documents trouvés avant reranking : {len(docs_to_rerank)}")
    
    if not docs_to_rerank:
        return []
        
    # 3. Reranking
    final_results = reranker.rerank(query_text, docs_to_rerank, top_k=RERANK_K)
    print(f"⭐ Top {len(final_results)} documents après reranking.")
    
    if return_dicts:
        return final_results
        
    # Retourner seulement le texte (pour compatibilité avec le code existant)
    return [doc["content"] for doc in final_results]

def clear_database():
    """
    Supprime physiquement le contenu du dossier ChromaDB et le dossier data/ pour repartir de zéro.
    """
    print("🧹 Purge de la base de données et des fichiers locaux...")
    
    # 1. Supprimer le contenu de ChromaDB (pas le dossier lui-même pour ne pas casser le volume Docker)
    chroma_abs_path = os.path.abspath(CHROMA_PATH)
    if os.path.exists(chroma_abs_path):
        for filename in os.listdir(chroma_abs_path):
            file_path = os.path.join(chroma_abs_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"❌ Erreur lors de la suppression de {file_path} : {e}")
        print(f"🗑️ Contenu de ChromaDB vidé : {chroma_abs_path}")
            
    # 2. Vider le dossier data/
    data_dir = os.path.join(PROJECT_ROOT, "data")
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
        
    # 3. Vider l'index BM25 en mémoire
    bm25_index.documents = []
    bm25_index.metadatas = []
    bm25_index.ids = []
    bm25_index.bm25 = None
        
    return True