import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const apiService = {
  askQuestion: async (text, history, config) => {
    const payload = {
      question: text,
      chat_history: history,
      top_k: 5,
      rerank_top_k: config.useReranker ? 3 : null,
      llm_params: {
        model: config.model,
        temperature: 0.7,
      },
    };
    const response = await axios.post(`${API_BASE_URL}/query`, payload);
    return response.data;
  },

  uploadFile: async (file) => {
    const formData = new FormData();
    formData.append("files", file);
    const response = await axios.post(`${API_BASE_URL}/ingest`, formData);
    return response.data;
  },
};
