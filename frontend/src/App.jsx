import React, { useState, useEffect, useCallback } from 'react';
import { Send, Upload, Database, ChevronLeft, ChevronRight, X, FileText, Play, Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Swal from 'sweetalert2';
import LogoAfriqueHub from './assets/logo-afrique-hub.jpeg';
import { apiService } from './services/api';

function App() {
  // --- ÉTATS ---
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState(null);

  // ÉTATS FICHIERS
  const [filesToUpload, setFilesToUpload] = useState([]); 
  const [uploadedFiles, setUploadedFiles] = useState([]); 
  const [isUploading, setIsUploading] = useState(false);

  // CONFIGURATION RAG
  const [selectedModel, setSelectedModel] = useState('Gemini 1.5 Flash');
  const [rerankEnabled, setRerankEnabled] = useState(true);

  // COPIER RÉPONSE
  const [copiedId, setCopiedId] = useState(null);
  const handleCopy = useCallback((id, content) => {
    navigator.clipboard.writeText(content).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    });
  }, []);

  //LOGIQUE D'UPLOAD 

  const handleFileSelection = (event) => {
    const selectedFiles = Array.from(event.target.files);
    setFilesToUpload(prev => [...prev, ...selectedFiles]);
    event.target.value = null; 
  };

  const removeFileFromQueue = (index) => {
    setFilesToUpload(prev => prev.filter((_, i) => i !== index));
  };

  const processIngestion = async () => {
    if (filesToUpload.length === 0) return;
    setIsUploading(true);
    let successCount = 0;

    try {
      for (const file of filesToUpload) {
        await apiService.uploadFile(file);
        setUploadedFiles(prev => [...prev, { name: file.name, size: file.size }]);
        successCount++;
      }

      Swal.fire({
        title: 'Documents Indexés !',
        text: `${successCount} fichiers ont été ajoutés à la base de connaissances.`,
        icon: 'success',
        confirmButtonColor: '#2563eb'
      });
      setFilesToUpload([]); 
    } catch (err) {
      Swal.fire({ title: 'Erreur', text: "Impossible d'indexer les documents.", icon: 'error' });
    } finally {
      setIsUploading(false);
    }
  };

  // LOGIQUE CHAT 

  /*const handleSendMessage = async () => {
    if (!input.trim() || isTyping) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setIsTyping(true);

    try {
      const data = await apiService.askQuestion(currentInput, messages, {
        model: selectedModel,
        useReranker: rerankEnabled
      });

      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer, 
        sources: data.sources 
      }]);
    } catch (err) {
      setError("Erreur de connexion. Veuillez réessayer.");
      setTimeout(() => setError(null), 5000);
    } finally {
      setIsTyping(false);
    }
  };*/
const [streamingId, setStreamingId] = useState(null)
const textareaRef = React.useRef(null);
const handleSendMessage = async () => {
  if (!input.trim() || isTyping) return;

  const userId = `user-${Date.now()}`;
  const assistantId = `assistant-${Date.now()}`;
  const currentInput = input;
  // Ne conserver que le dernier message de l'assistant et le dernier message de l'utilisateur
  const historySnapshot = [];
  const rev = [...messages].slice().reverse();
  const lastAssistant = rev.find(m => m.role === 'assistant');
  const lastUser = rev.find(m => m.role === 'user');
  // On suit l'ordre demandé: assistant puis utilisateur
  if (lastAssistant) historySnapshot.push({ role: 'assistant', content: lastAssistant.content });
  if (lastUser) historySnapshot.push({ role: 'user', content: lastUser.content });

  // 1. Reset UI
  setInput('');
  if (textareaRef.current) textareaRef.current.style.height = 'auto';
  setIsTyping(true);

  // 2. Ajouter le message utilisateur
  setMessages(prev => [...prev, { id: userId, role: 'user', content: currentInput }]);

  try {
    const data = await apiService.askQuestion(currentInput, historySnapshot, {
      model: selectedModel,
      useReranker: rerankEnabled
    });

    // 3. Créer la bulle assistant VIDE
    setMessages(prev => [...prev, { 
      id: assistantId, 
      role: 'assistant', 
      content: '', 
      sources: data.sources 
    }]);


    // 4. Simuler le streaming en affichant progressivement la réponse
    const fullResponse = data.answer;

    setStreamingId(assistantId);

      for (let i = 0; i < fullResponse.length; i++) {
      const char = fullResponse[i];

      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantId
            ? { ...msg, content: msg.content + char }
            : msg
        )
      );

      await new Promise(r => setTimeout(r, 18));
    }

    setIsTyping(false);
    setStreamingId(null);

  } catch (err) {
    console.error(err);
    setError("Erreur de connexion.");
    setIsTyping(false);
  }
};

const messagesEndRef = React.useRef(null);
useEffect(() => {
  messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
}, [messages]);
  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 overflow-hidden relative font-sans">
      
      {/* BOUTON TOGGLE SIDEBAR */}
      <button 
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
        className={`fixed top-3 z-50 p-2 bg-blue-600 text-white rounded-full shadow-xl transition-all duration-300 hover:bg-blue-700
          ${isSidebarOpen ? 'left-72' : 'left-4'}`}
      >
        {isSidebarOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
      </button>

      {/* OVERLAY MOBILE */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/30 z-30 lg:hidden backdrop-blur-sm" 
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* SIDEBAR */}
      <aside className={`
        fixed inset-y-0 left-0 z-40 w-80 bg-white border-r border-slate-200 p-6 transform transition-transform duration-300 ease-in-out
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        lg:relative lg:translate-x-0 ${!isSidebarOpen && 'lg:hidden'}
      `}>
        <div className="flex items-center gap-2 font-bold text-xl text-blue-600 mb-8">
          <img src={LogoAfriqueHub} alt="Logo" className="w-10 h-10 object-contain" />
          <span className="truncate">RAG Data Afrique Hub</span>
        </div>

        <div className="space-y-6 overflow-y-auto max-h-[calc(100vh-120px)] pr-2 custom-scrollbar">
          
          {/* SECTION UPLOAD */}
          <div>
            <h3 className="text-xs font-bold uppercase text-slate-400 mb-3 tracking-wider">Documents</h3>
            <label className="flex flex-col items-center justify-center w-full h-24 border-2 border-dashed border-slate-200 rounded-2xl cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-all mb-4">
              <Upload className="text-slate-400 mb-1" size={20} />
              <span className="text-[11px] text-slate-500 font-medium">Glissez vos fichiers ici</span>
              <input type="file" className="hidden" multiple onChange={handleFileSelection} disabled={isUploading} />
            </label>

            {/* File d'attente (Orange) */}
            {filesToUpload.length > 0 && (
              <div className="mb-4 animate-in slide-in-from-bottom-2">
                <p className="text-[10px] font-bold text-orange-600 uppercase mb-2 flex justify-between">
                  À envoyer <span>{filesToUpload.length}</span>
                </p>
                <div className="space-y-1 max-h-32 overflow-y-auto mb-2">
                  {filesToUpload.map((file, idx) => (
                    <div key={idx} className="flex items-center gap-2 p-2 bg-orange-50 border border-orange-100 rounded-lg text-[11px]">
                      <FileText size={14} className="text-orange-400" />
                      <span className="truncate flex-1 font-medium">{file.name}</span>
                      <X size={14} className="text-slate-400 hover:text-red-500 cursor-pointer" onClick={() => removeFileFromQueue(idx)} />
                    </div>
                  ))}
                </div>
                <button 
                  onClick={processIngestion}
                  disabled={isUploading}
                  className="w-full py-2 bg-blue-600 text-white rounded-xl text-xs font-bold flex items-center justify-center gap-2 hover:bg-blue-700 transition-all"
                >
                  {isUploading ? "En cours..." : <><Play size={12}/> Indexer dans la base</>}
                </button>
              </div>
            )}

            {/* Déjà Indexés (Gris) */}
            {uploadedFiles.length > 0 && (
              <div className="pt-4 border-t border-slate-100">
                <p className="text-[10px] font-bold text-slate-400 uppercase mb-2">Base de données</p>
                <div className="space-y-1">
                  {uploadedFiles.map((f, i) => (
                    <div key={i} className="flex items-center gap-2 p-2 bg-slate-50 rounded-lg text-[10px] text-slate-500 italic">
                      <Database size={12} className="text-blue-400"/>
                      <span className="truncate">{f.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* CONFIGURATION */}
          <div className="pt-4 border-t border-slate-100">
            <h3 className="text-xs font-bold uppercase text-slate-400 mb-3 tracking-wider">Modèle IA</h3>
            <select 
              className="w-full p-2.5 bg-slate-50 border border-slate-200 rounded-xl text-xm  focus:ring-2 focus:ring-blue-500 outline-none"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="Gemini 1.5 Flash">Gemini 1.5 Flash</option>
              <option value="GPT-4o-mini">GPT-4o-mini</option>
              <option value="Ollama-Llama3">Llama 3 (Local)</option>
              <option value="Mistral-7B">Mistral 7B (Local/Gratuit)</option>
            </select>
            
            <div className="mt-3 flex items-center gap-2 p-2.5 bg-slate-50 rounded-xl border border-slate-100">
              <input type="checkbox" id="rerank" checked={rerankEnabled} onChange={(e) => setRerankEnabled(e.target.checked)} className="accent-blue-600" />
              <label htmlFor="rerank" className="text-sm font-medium text-slate-600 cursor-pointer">Re-ranking activé</label>
            </div>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 flex flex-col min-w-0 bg-white lg:bg-slate-50 relative">
        {/* HEADER */}
        <header className={`
          h-16 bg-white border-b border-slate-200 flex items-center transition-all duration-300 shadow-sm
          ${isSidebarOpen ? 'pl-16 lg:pl-8' : 'pl-16'}
        `}>
          <h1 className="font-semibold text-slate-700 truncate text-sm md:text-base">ASSISTANT POUR REPONDRE A VOS QUESTIONS</h1>
        </header>

        {/* ERREUR */}
        {error && (
          <div className="mx-4 mt-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg flex items-center justify-between text-xs animate-bounce">
            <span>{error}</span>
            <X size={14} className="cursor-pointer" onClick={() => setError(null)} />
          </div>
        )}

        {/* ZONE DE CHAT */}
        <section className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 custom-scrollbar">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center opacity-30">
              <Database size={60} className="mb-4 text-blue-600" />
              <p className="text-lg font-medium">Posez une question sur vos documents</p>
            </div>
          )}
          
          {messages.map((m, i) => (
            <div key={m.id} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2`}>
             {/* <div className={`max-w-[90%] md:max-w-2xl p-4 rounded-2xl shadow-sm text-sm leading-relaxed
                ${m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border border-slate-200'}`}>
                {m.content}*/}
              <div className={`max-w-[90%] md:max-w-2xl p-4 rounded-2xl shadow-sm text-sm leading-relaxed
                ${m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border border-slate-200'}`}>
                <div className="flex items-start gap-3">
                  {/* Icône IA */}
                  {m.role === 'assistant' && (
                    <div className="mt-1 p-1 rounded-full bg-slate-100 shrink-0">
                      <Database size={16} className="text-blue-600" />
                    </div>
                  )}

                  <div className="flex-1 min-w-0">
                    {m.role === 'assistant' ? (
                      <>
                        <div className="markdown">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {m.content}
                          </ReactMarkdown>
                        </div>
                        {/* Curseur streaming */}
                        {m.id === streamingId && (
                          <span className="inline-block w-2 h-4 ml-1 bg-blue-400 animate-pulse align-middle" />
                        )}
                      </>
                    ) : (
                      <div className="whitespace-pre-wrap">{m.content}</div>
                    )}
                  </div>

                  {/* Bouton copier (assistant seulement, hors streaming) */}
                  {m.role === 'assistant' && m.id !== streamingId && m.content && (
                    <button
                      onClick={() => handleCopy(m.id, m.content)}
                      className="shrink-0 mt-1 p-1 rounded text-slate-300 hover:text-slate-500 transition-colors"
                      title="Copier la réponse"
                    >
                      {copiedId === m.id
                        ? <Check size={14} className="text-green-500" />
                        : <Copy size={14} />}
                    </button>
                  )}
                </div>

                {/* Sources avec score */}
                {m.sources && m.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-slate-100 flex flex-wrap gap-1.5">
                    {m.sources.map((source, idx) => {
                      const score = source.metadata?.score ?? source.score;
                      const pct = score != null ? Math.round(score * 100) : null;
                      return (
                        <div key={idx} className="flex items-center gap-1 bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full text-[9px] font-bold">
                          <FileText size={9} />
                          <span>{source.metadata?.filename || source.metadata?.source || "Source"}</span>
                          {pct != null && (
                            <span className="text-blue-400 font-normal">{pct}%</span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div> 
              
            </div>
          ))}
          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-white border border-slate-200 p-4 rounded-2xl flex gap-2 items-center text-slate-400 text-xs">
                <span className="flex gap-1"><span className="w-1 h-1 bg-slate-400 rounded-full animate-bounce"></span><span className="w-1 h-1 bg-slate-400 rounded-full animate-bounce [animation-delay:0.2s]"></span><span className="w-1 h-1 bg-slate-400 rounded-full animate-bounce [animation-delay:0.4s]"></span></span>
                Analyse en cours...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </section>

        {/* FOOTER / INPUT */}
        <footer className="p-4 bg-white border-t border-slate-200">
          <div className="max-w-4xl mx-auto flex gap-2 items-center bg-slate-100 p-2 rounded-2xl border border-slate-200 focus-within:ring-2 focus-within:ring-blue-400 transition-all">
            <textarea
              ref={textareaRef} 
              className="flex-1 p-2 bg-transparent outline-none text-sm resize-none max-h-32"
              placeholder="Écrivez votre message..."
              rows="1"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = 'auto';
                e.target.style.height = `${e.target.scrollHeight}px`;
              }}
              onKeyDown={(e) => {
                if(e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
            />
            <button 
              onClick={handleSendMessage}
              disabled={isTyping}
              className="p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 active:scale-95 transition-all disabled:opacity-50"
            >
              <Send size={18} />
            </button>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App; 