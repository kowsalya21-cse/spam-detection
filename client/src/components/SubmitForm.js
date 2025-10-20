// client/src/components/SubmitForm.js
import React, { useState, useEffect } from 'react';
import { postMessage, getMessages } from '../api';

export default function SubmitForm(){
  const [text, setText] = useState('');
  const [messages, setMessages] = useState([]);

  async function load(){
    const res = await getMessages();
    setMessages(res.data);
  }
  useEffect(()=> { load(); }, []);

  async function handleSend(e){
    e.preventDefault();
    if(!text) return;
    const res = await postMessage(text);
    setText('');
    load();
  }

  return (
    <div style={{maxWidth:700, margin:'20px auto'}}>
      <h3>SMS Spam Demo</h3>
      <form onSubmit={handleSend}>
        <textarea value={text} onChange={e=>setText(e.target.value)} rows={3} style={{width:'100%'}}/>
        <button type="submit">Send & Classify</button>
      </form>

      <h4>Last messages</h4>
      <ul>
        {messages.slice(0,20).map(m=>(
          <li key={m._id}>
            <b>{m.label===1 ? 'SPAM' : 'HAM'}</b> — {m.text} <small>({(m.prob||0).toFixed(2)})</small>
          </li>
        ))}
      </ul>
    </div>
  );
}
