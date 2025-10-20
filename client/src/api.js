// client/src/api.js
import axios from 'axios';
const BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000';
export const postMessage = (text) => axios.post(`${BASE}/api/messages`, { text });
export const getMessages = () => axios.get(`${BASE}/api/messages`);
