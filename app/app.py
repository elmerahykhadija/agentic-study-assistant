import streamlit as st
import sys
import os

# Ajout du chemin racine pour permettre les imports des modules backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.graph import app as langgraph_app
from vectorstore.chroma_client import clear_database

# --- Configuration de la page ---
st.set_page_config(page_title="Agentic Study Assistant", page_icon="🎓", layout="wide")

# --- Initialisation de la mémoire de chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# BARRE LATÉRALE : INGESTION DE DONNÉES ET GESTION
# ==========================================
with st.sidebar:
    st.title("🎓 Assistant d'Étude")
    st.markdown("---")
    
    st.header("📂 Base de connaissances")
    st.write("Ajoutez vos cours avant de poser des questions.")
    
    # Option Lien Drive
    drive_link = st.text_input("🔗 Lier un dossier Google Drive public")
    if st.button("Vectoriser le Drive", use_container_width=True):
        if drive_link:
            with st.spinner("Téléchargement, OCR et vectorisation en cours..."):
                inputs = {"user_input": drive_link, "input_type": "lien"}
                try:
                    for output in langgraph_app.stream(inputs):
                        for key, value in output.items():
                            if "final_answer" in value:
                                st.success(value["final_answer"])
                except Exception as e:
                    st.error(f"Erreur d'ingestion : {e}")
        else:
            st.warning("Veuillez entrer un lien valide.")
            
    st.markdown("---")
    
    # Option Fichier Local
    uploaded_file = st.file_uploader("📄 Uploader un document", type=["pdf", "png", "jpg"])
    if st.button("Vectoriser le fichier", use_container_width=True):
        if uploaded_file:
            st.info("Le parseur de document local sera bientôt connecté à ce bouton !")
        else:
            st.warning("Veuillez sélectionner un fichier.")
            
    st.markdown("---")
    
    # Section de Nettoyage
    st.header("⚙️ Gestion")
    
    if st.button("🔄 Nouvelle Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.write("") # Espace
    
    if st.button("🗑️ Purger la base de connaissances", type="primary", use_container_width=True, help="Supprime tous les documents stockés de la base de données."):
        with st.spinner("Suppression de la base de données..."):
            clear_database()
            st.session_state.messages = [] # On vide aussi le chat pour éviter les hallucinations
            st.success("Base de connaissances et fichiers locaux purgés avec succès !")

# ==========================================
# ZONE PRINCIPALE : CHAT & AGENTS
# ==========================================
st.title("Discutez avec vos cours")
st.markdown("Posez vos questions ou demandez la génération de schémas explicatifs.")

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Saisie de l'utilisateur
if prompt := st.chat_input("Demandez une explication, un résumé ou un schéma..."):
    
    # 1. Afficher la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Lancer le graphe agentique
    with st.chat_message("assistant"):
        with st.spinner("L'agent réfléchit (Évaluation CRAG & Génération)..."):
            # L'historique transmis correspond à tous les messages SAUF la requête actuelle.
            history = st.session_state.messages[:-1]
            inputs = {
                "user_input": prompt, 
                "input_type": "question",
                "chat_history": history
            }
            final_response = "Une erreur est survenue lors de la réflexion."
            
            try:
                # Parcours des nœuds du graphe
                for output in langgraph_app.stream(inputs):
                    for key, value in output.items():
                        # On intercepte la réponse des nœuds finaux (générateurs)
                        if "final_answer" in value:
                            final_response = value["final_answer"]
                
                st.markdown(final_response)
                
                # Sauvegarde dans l'historique
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
            except Exception as e:
                st.error(f"Erreur d'exécution du pipeline : {e}")