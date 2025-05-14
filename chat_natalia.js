const btn       = document.getElementById('chat-button');
const container = document.getElementById('chat-container');
const closeBtn  = document.getElementById('close-btn');
const box       = document.getElementById('chat-box');
const input     = document.getElementById('chat-input');
const send      = document.getElementById('send-btn');

const sessionId = 'chat_' + Date.now();

let lastMessageCount = 0;
let isFirstLoad = true;

// Show/hide chat window
btn.onclick = () => {
  container.classList.toggle('hidden');
  if (!container.classList.contains('hidden')) {
    box.innerHTML = '';
    lastMessageCount = 0;
    isFirstLoad = true;
    setTimeout(() => {
      input.focus();
      fetchMessages();
    }, 150);
  }
};
closeBtn.onclick = () => {
  container.classList.add('hidden');
  setTimeout(() => { box.innerHTML = ''; }, 200);
};

// Append message to chat
function append(who, text) {
  const div = document.createElement('div');
  div.className = 'message ' + who;
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

// Send message to backend
async function postMessage(text) {
  const userMsg = append('user', text);
  input.value = '';

  const loader = append('bot', '...');

  const payload = {
    url: window.location.origin,
    max_pages: 200,
    model: 'gpt-4.1-mini',
    question: text,
    conversation_id: sessionId
  };

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.answer) {
      loader.textContent = data.answer;
      // Persist escalation messages
      if (data.type === 'escalation') {
        loader.classList.add('persist', 'escalation');
      }
    }
    lastMessageCount += 2; // One for user message, one for response
  } catch (e) {
    loader.textContent = '⚠️ Error: ' + e.message;
    lastMessageCount++;
  }
}

// Fetch all messages from Flask
async function fetchMessages() {
  try {
    const res = await fetch(`/api/messages/${sessionId}`);
    const messages = await res.json();
    
    if (!messages || messages.length === 0) {
      return;
    }

    // Only remove non-persistent loading indicators
    const loaders = box.querySelectorAll('.message.bot:not(.persist)');
    loaders.forEach(loader => {
      if (loader.textContent === '...') {
        loader.remove();
      }
    });

    // Reset messages if count decreased (new session)
    if (messages.length < lastMessageCount) {
      box.innerHTML = '';
      lastMessageCount = 0;
    }

    // Add new messages
    if (messages.length > lastMessageCount) {
      const newMessages = messages.slice(lastMessageCount);
      newMessages.forEach(msg => {
        append(msg.role === 'user' ? 'user' : 'bot', msg.content);
      });
      lastMessageCount = messages.length;
    }
  } catch (err) {
    console.error("Polling error:", err);
  }
}

// Hook up send
send.onclick = () => {
  if (input.value.trim()) postMessage(input.value);
};
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send.click();
  }
});

// Poll for new messages every 200ms
setInterval(fetchMessages, 200);
fetchMessages();
