# 🎓 Agentic Study Assistant : Architecture CRAG & LLMs Multimodaux

## 📌 Contexte
Dans les cycles d'ingénierie avancés, particulièrement en Systèmes d'Information et Big Data, le volume de documentation académique et technique (polycopiés complexes, schémas d'architecture, TD, TP) est massif. Les étudiants perdent un temps précieux à rechercher des informations spécifiques à travers des centaines de pages non structurées.

## ⚠️ Problématique
- **Surcharge d'information** : La recherche manuelle dans des dossiers cloud massifs est inefficace.
- **Hallucination des LLMs** : Les modèles génératifs standards (ChatGPT, Claude) inventent souvent des réponses lorsqu'ils sont interrogés sur des cours spécifiques et privés.
- **Limites du RAG Classique** : Un système RAG (Retrieval-Augmented Generation) standard échoue silencieusement si la recherche vectorielle renvoie des données peu pertinentes, générant ainsi une réponse basée sur un mauvais contexte.

## 🎯 Objectif
Concevoir un pipeline agentique de bout en bout capable d'ingérer de larges corpus de cours (depuis Google Drive ou en local), d'extraire intelligemment le texte (y compris via OCR pour les schémas scannés), et de fournir des réponses fiables à 100%. Le système doit être multimodal : capable d'expliquer des concepts sous forme de texte, de générer des diagrammes d'architecture, ou de consolider le savoir dans des rapports PDF dynamiques, tout en intégrant un filet de sécurité anti-hallucination.

## 🧠 Pourquoi CRAG et pas un RAG classique ?
Dans un RAG classique, le flux est linéaire et aveugle : `Question ➔ Recherche Vectorielle ➔ Génération`.

**Le problème** : Si le système vectoriel ne trouve pas la bonne information (ou trouve un paragraphe ambigu), le LLM essaiera quand même de répondre, ce qui provoque des hallucinations.

Dans cette architecture **CRAG (Corrective RAG)**, un agent Évaluateur est inséré au cœur du système :
`Question ➔ Recherche Vectorielle ➔ ÉVALUATION STRICTE ➔ (Routage dynamique) ➔ Génération`.

**La solution** : L'évaluateur note le contexte extrait (de 0.0 à 1.0) ou l'évalue catégoriquement (correct, ambigu, incorrect).
- Si le score est bon, le contexte est approuvé pour la génération.
- Si le score est insuffisant (Ambigu ou Incorrect), le système rejette le contexte interne et déclenche de manière autonome un outil de Recherche Web (DuckDuckGo).

**Avantage majeur** : Ce mécanisme garantit une tolérance zéro aux hallucinations, une qualité indispensable pour un système d'IA déployé en production.

## ⚙️ Explication des Composants (Architecture Agentique)
L'application repose sur un écosystème d'agents spécialisés orchestrés par LangGraph, interagissant avec une mémoire vectorielle.

- **`vectorstore/chroma_client.py`** : Le moteur de la base de données vectorielle. Il gère l'embedding des chunks de texte (via `paraphrase-multilingual-MiniLM-L12-v2` pour la performance) et permet des recherches de similarité ultra-rapides.
- **`ingestion/pipeline.py` & `tools/document_parser.py`** : Le pipeline de Data Engineering. Utilise PyMuPDF pour l'extraction de texte et Tesseract OCR pour déchiffrer les schémas et images scannées, suivi d'un découpage sémantique intelligent (LangChain Recursive Text Splitter).
- **`agent/evaluator.py`** : L'agent critique du CRAG. Propulsé par Llama-3, il évalue mathématiquement la pertinence du contexte extrait par rapport à la question.
- **`tools/web_search.py`** : L'outil de secours (Fallback). Intègre un optimiseur de requête SEO (LLM) avant d'interroger le web via DuckDuckGo.
- **`agent/generator_router.py`** : Le classifieur d'intention. Il analyse la question de l'utilisateur pour comprendre le format de sortie désiré et l'oriente vers le bon pipeline de génération (Texte, Visuel, PDF).
- **Les Générateurs (`text_generator.py`, `visual_generator.py`, `pdf_generator.py`)** : Les nœuds finaux qui créent les livrables. Ils gèrent la synthèse Markdown, la génération de code Graphviz (DOT) pour les diagrammes d'architecture, et la compilation WeasyPrint pour les rapports HTML/PDF professionnels.
- **`app/app.py`** : L'interface utilisateur interactive construite avec Streamlit, gérant l'historique de conversation (State) et la restitution visuelle des artefacts générés.

## 🔄 Enchaînement du Workflow (Graphe LangGraph)
L'orchestration est modélisée sous forme de graphe d'états (StateGraph). Voici le cycle de vie exact d'une requête :

1. **Routage Principal (`orchestrator_router`)** :
   L'utilisateur soumet une requête. L'orchestrateur vérifie son type.
   - S'il s'agit d'une URL Drive ou d'un fichier, le flux part vers l'ingestion (OCR ➔ Chunking ➔ ChromaDB) et s'arrête.
   - S'il s'agit d'une question, le flux part vers le nœud de récupération (Retrieval).

2. **Récupération & Évaluation (`retrieve_node` ➔ `evaluate_node`)** :
   Interrogation de ChromaDB. Le contexte est extrait puis envoyé à l'évaluateur CRAG.

3. **Routage CRAG (`crag_router`)** :
   L'évaluateur rend son verdict. Si le contexte est fiable, passage à l'étape suivante. Si le contexte est insuffisant, détournement vers le `web_search_node` pour enrichissement en temps réel.

4. **Analyse d'Intention (`route_generator_node`)** :
   L'agent détermine le format de sortie attendu (Texte standard, Schéma PNG/PDF, Rapport PDF complexe).

5. **Génération & Restitution** :
   Le flux atteint le générateur approprié (`generate_text`, `generate_visual`, ou `generate_pdf`). L'agent final compile la réponse en respectant un prompt strict d'anti-hallucination et d'intégration de l'historique de conversation, puis renvoie le livrable à l'interface Streamlit.

## 🚀 Déploiement et Utilisation
L'ensemble de l'environnement est conteneurisé pour assurer une reproductibilité parfaite.

**Prérequis** : Docker et Docker Compose.

1. Clonez le dépôt et insérez vos identifiants dans le fichier d'environnement :
   ```bash
   cp infra/.env.example infra/.env
   # Ajoutez votre clé API Groq (GROQ_API_KEY)
   ```

2. Construisez et lancez l'application via Docker :
   ```bash
   cd infra
   sudo docker-compose up --build -d
   ```

3. Accédez à l'interface utilisateur Streamlit via votre navigateur web :
   [http://localhost:8501](http://localhost:8501)
