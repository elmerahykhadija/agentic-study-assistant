import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import {
  Send,
  UploadCloud,
  FolderSync,
  Trash2,
  Menu,
  GraduationCap,
  Bot,
  User,
  Loader2,
  RefreshCcw,
  FileText,
  MessageSquare,
  X
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [driveUrl, setDriveUrl] = useState('');
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const messagesEndRef = useRef(null);

  // Historique des chats
  const [chatHistory, setChatHistory] = useState(() => {
    const saved = localStorage.getItem('study_assistant_chats');
    return saved ? JSON.parse(saved) : {};
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Sauvegarde de l'historique
  useEffect(() => {
    if (messages.length > 0) {
      setChatHistory(prev => {
        const newHistory = { ...prev };
        newHistory[sessionId] = {
          id: sessionId,
          messages: messages,
          title: messages[0]?.content.slice(0, 30) + '...',
          updatedAt: Date.now()
        };
        localStorage.setItem('study_assistant_chats', JSON.stringify(newHistory));
        return newHistory;
      });
    }
  }, [messages, sessionId]);

  const loadSession = (id) => {
    const session = chatHistory[id];
    if (session) {
      setSessionId(id);
      setMessages(session.messages);
      if (window.innerWidth < 768) setSidebarOpen(false);
    }
  };

  const deleteSession = (id, e) => {
    e.stopPropagation();
    setChatHistory(prev => {
      const newHistory = { ...prev };
      delete newHistory[id];
      localStorage.setItem('study_assistant_chats', JSON.stringify(newHistory));
      return newHistory;
    });
    if (sessionId === id) {
      startNewConversation();
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/chat`, {
        session_id: sessionId,
        message: input,
        chat_history: messages,
        web_search_enabled: webSearchEnabled
      });

      setMessages(prev => [...prev, { role: 'assistant', content: response.data.response }]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages(prev => [...prev, { role: 'assistant', content: '❌ Erreur de connexion avec le serveur.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    try {
      const response = await axios.post(`${API_BASE}/ingest/file`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      alert(response.data.message);
      setUploadedFiles(prev => [...prev, file.name]);
    } catch (error) {
      alert("Erreur lors de l'upload du fichier.");
    } finally {
      setIsUploading(false);
      e.target.value = null;
    }
  };

  const handleDriveIngest = async () => {
    if (!driveUrl) {
      alert("Veuillez entrer une URL valide.");
      return;
    }

    setIsUploading(true);
    try {
      const response = await axios.post(`${API_BASE}/ingest/drive`, {
        session_id: sessionId,
        drive_url: driveUrl
      });
      alert(response.data.message);
      setUploadedFiles(prev => [...prev, driveUrl]);
      setDriveUrl('');
    } catch (error) {
      alert("Erreur lors de l'ingestion Google Drive.");
    } finally {
      setIsUploading(false);
    }
  };

  const startNewConversation = () => {
    setMessages([]);
    setSessionId(crypto.randomUUID());
  };

  const clearDatabase = async () => {
    if (!confirm("Êtes-vous sûr de vouloir purger toute la base de connaissances ? Cela n'effacera pas votre historique de chat local.")) return;
    try {
      await axios.post(`${API_BASE}/clear`, { session_id: sessionId });
      alert("Base purgée avec succès.");
      setUploadedFiles([]);
    } catch (error) {
      alert("Erreur lors de la purge.");
    }
  };

  // Tri de l'historique par date décroissante
  const sortedHistory = Object.values(chatHistory).sort((a, b) => b.updatedAt - a.updatedAt);

  return (
    <div className="flex h-screen bg-gray-50 text-gray-800 font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className={`bg-white border-r border-gray-200 transition-all duration-300 flex flex-col ${sidebarOpen ? 'w-80' : 'w-0'}`}>
        <div className="flex flex-col h-full w-80">
          <div className="p-6 flex items-center gap-3 border-b border-gray-100 shrink-0">
            <div className="bg-purple-600 p-2 rounded-xl text-white">
              <GraduationCap size={24} />
            </div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
              Study Assistant
            </h1>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-8">
            <section>
              <button
                onClick={startNewConversation}
                className="w-full flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium py-3 px-4 rounded-lg transition-colors mb-6 shadow-sm"
              >
                <MessageSquare size={18} /> Nouvelle Conversation
              </button>

              <h2 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4">Base de Connaissances</h2>

              <div className="space-y-4">
                <div className="bg-gray-50 p-4 rounded-xl border border-gray-100">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Google Drive Public</label>
                  <input
                    type="text"
                    placeholder="Lien URL..."
                    className="w-full px-3 py-2 text-sm bg-white border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 mb-3"
                    value={driveUrl}
                    onChange={(e) => setDriveUrl(e.target.value)}
                  />
                  <button
                    onClick={handleDriveIngest}
                    disabled={isUploading}
                    className="w-full flex items-center justify-center gap-2 bg-white border border-gray-200 hover:bg-gray-50 disabled:opacity-50 text-gray-700 text-sm font-medium py-2 px-4 rounded-lg transition-colors"
                  >
                    {isUploading ? <Loader2 size={16} className="animate-spin text-purple-600" /> : <FolderSync size={16} className="text-purple-600" />}
                    {isUploading ? 'Vectorisation...' : 'Vectoriser Drive'}
                  </button>
                </div>

                <div className="bg-gray-50 p-4 rounded-xl border border-gray-100">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Fichier Local</label>
                  <div className="relative">
                    <input
                      type="file"
                      id="file-upload"
                      className="hidden"
                      onChange={handleFileUpload}
                      accept=".pdf,.doc,.docx"
                      disabled={isUploading}
                    />
                    <label
                      htmlFor="file-upload"
                      className={`w-full flex flex-col items-center justify-center gap-2 bg-white border border-dashed border-gray-300 hover:border-purple-500 hover:bg-purple-50 text-gray-500 text-sm font-medium py-4 px-4 rounded-lg ${isUploading ? 'cursor-not-allowed opacity-70' : 'cursor-pointer'} transition-colors`}
                    >
                      {isUploading ? (
                        <>
                          <Loader2 size={20} className="animate-spin text-purple-500" />
                          <span>Upload en cours...</span>
                        </>
                      ) : (
                        <>
                          <UploadCloud size={20} className="text-purple-500" />
                          <span>Uploader un document</span>
                          <span className="text-xs text-gray-400">PDF, DOC, DOCX</span>
                        </>
                      )}
                    </label>
                  </div>

                  {uploadedFiles.length > 0 && (
                    <div className="mt-4 space-y-2">
                      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Fichiers indexés</h3>
                      {uploadedFiles.map((fname, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm text-gray-700 bg-white p-2 rounded border border-gray-200">
                          <FileText size={14} className="text-purple-500 shrink-0" />
                          <span className="truncate flex-1" title={fname}>{fname}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </section>

            {sortedHistory.length > 0 && (
              <section>
                <h2 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4 flex justify-between items-center">
                  Historique
                  <span className="bg-gray-100 text-gray-500 py-0.5 px-2 rounded-full text-[10px]">{sortedHistory.length}</span>
                </h2>
                <div className="space-y-1">
                  {sortedHistory.map((session) => (
                    <div
                      key={session.id}
                      onClick={() => loadSession(session.id)}
                      className={`flex items-center justify-between p-2.5 rounded-lg cursor-pointer transition-colors group ${
                        sessionId === session.id ? 'bg-purple-50 text-purple-700 border border-purple-100' : 'hover:bg-gray-100 text-gray-600 border border-transparent'
                      }`}
                    >
                      <div className="flex items-center gap-3 overflow-hidden">
                        <MessageSquare size={14} className={sessionId === session.id ? 'text-purple-500 shrink-0' : 'text-gray-400 shrink-0'} />
                        <div className="truncate text-sm font-medium">{session.title}</div>
                      </div>
                      <button
                        onClick={(e) => deleteSession(session.id, e)}
                        className="text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity p-1"
                        title="Supprimer"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section>
              <h2 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4">Paramètres</h2>
              <div className="space-y-3">
                <label className="flex items-center gap-3 text-sm text-gray-700 cursor-pointer bg-white p-3 rounded-lg border border-gray-200 transition-colors hover:bg-gray-50">
                  <input
                    type="checkbox"
                    className="rounded text-purple-600 focus:ring-purple-500 w-4 h-4"
                    checked={webSearchEnabled}
                    onChange={(e) => setWebSearchEnabled(e.target.checked)}
                  />
                  <span>Autoriser la recherche Web</span>
                </label>
                <button
                  onClick={clearDatabase}
                  className="w-full flex items-center gap-3 bg-white hover:bg-red-50 text-red-600 border border-gray-200 hover:border-red-200 text-sm font-medium py-2.5 px-4 rounded-lg transition-colors"
                >
                  <Trash2 size={16} /> Purger la base de données
                </button>
              </div>
            </section>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full bg-white relative">
        <header className="h-16 flex items-center px-4 border-b border-gray-100 bg-white/80 backdrop-blur-md absolute top-0 w-full z-10 shrink-0">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 text-gray-500 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <Menu size={20} />
          </button>
          <div className="ml-4">
            <h2 className="text-sm font-bold text-gray-800">Chat avec l'Assistant</h2>
            <p className="text-xs text-gray-500">ID Session: <span className="font-mono text-purple-600 bg-purple-50 px-1 rounded">{sessionId.split('-')[0]}</span></p>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto pt-20 pb-24 px-4 sm:px-8">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto">
              <div className="bg-purple-100 p-5 rounded-full mb-6">
                <Bot size={48} className="text-purple-600" />
              </div>
              <h2 className="text-3xl font-bold text-gray-800 mb-3 tracking-tight">Bonjour ! 👋</h2>
              <p className="text-gray-500 text-lg">Je suis votre assistant d'étude propulsé par l'IA. Ajoutez des cours dans la barre latérale, puis posez-moi vos questions.</p>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((msg, index) => (
                <div key={index} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1 ${msg.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-purple-100 text-purple-600'}`}>
                    {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                  </div>
                  <div className={`px-5 py-4 rounded-2xl max-w-[85%] shadow-sm overflow-hidden ${msg.role === 'user'
                      ? 'bg-indigo-600 text-white rounded-tr-sm'
                      : 'bg-white text-gray-800 rounded-tl-sm border border-gray-200'
                    }`}>
                    <div className={`prose max-w-none leading-relaxed text-[15px] ${msg.role === 'user' ? 'prose-invert' : 'prose-purple'}`}>
                      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center shrink-0 mt-1">
                    <Loader2 size={16} className="animate-spin" />
                  </div>
                  <div className="px-5 py-4 rounded-2xl bg-white border border-gray-200 text-gray-500 shadow-sm rounded-tl-sm flex items-center gap-2">
                    L'agent réfléchit...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="absolute bottom-0 w-full bg-gradient-to-t from-white via-white to-transparent pt-8 pb-6 px-4 sm:px-8">
          <form onSubmit={handleSendMessage} className="max-w-3xl mx-auto relative group flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Posez votre question sur le cours..."
              disabled={isLoading}
              className="flex-1 bg-white border border-gray-300 shadow-sm rounded-full py-4 pl-6 pr-4 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-200 text-gray-800 disabled:bg-gray-50 transition-all text-base"
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="w-14 h-14 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-200 disabled:text-gray-400 text-white rounded-full flex items-center justify-center transition-colors shadow-sm shrink-0"
            >
              <Send size={18} className={input.trim() && !isLoading ? 'ml-0.5' : ''} />
            </button>
          </form>
          <div className="text-center mt-3">
            <span className="text-xs text-gray-400">Agentic Study Assistant génère ses réponses à partir des documents fournis.</span>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
