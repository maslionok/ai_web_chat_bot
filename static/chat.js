const btn       = document.getElementById('chat-button');
const container = document.getElementById('chat-container');
const closeBtn  = document.getElementById('close-btn');
const box       = document.getElementById('chat-box');
const input     = document.getElementById('chat-input');
const send      = document.getElementById('send-btn');

// Toggle chat visibility
btn.onclick = () => {
  container.classList.toggle('hidden');
  if (!container.classList.contains('hidden')) {
    setTimeout(() => input.focus(), 150);
  }
};
closeBtn.onclick = () => {
  container.classList.add('hidden');
  // Optional: clear chat on close for a fresh start
  setTimeout(() => { box.innerHTML = ''; }, 200);
};

// Store conversation history
const conversationHistory = [];

// Append a message; returns the div
function append(who, text) {
  const div = document.createElement('div');
  div.className   = 'message ' + who;
  div.textContent = text;
  div.style.opacity = 0;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  setTimeout(() => { div.style.transition = 'opacity 0.25s'; div.style.opacity = 1; }, 10);
  conversationHistory.push({ who, text }); // Save to history
  return div;
}

// Send a message to the backend AI
async function postMessage(text) {
  append('user', text);
  input.value = '';

  // show loading dots
  const loader = append('bot', '...');

  // Use default website and page limit
  const payload = {
    url: window.location.origin,
    max_pages: 200,
    model: 'gpt-4.1-mini',
    question: text,
    history: conversationHistory // Include conversation history
  };

  try {
    const res = await fetch('/api/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });
    const resJSON = await res.json();
    if (resJSON.crawl_done) {
      btn.style.background = "linear-gradient(135deg, green 60%, darkgreen 100%)";
    }
    loader.textContent = resJSON.answer;
    conversationHistory.push({ who: 'bot', text: resJSON.answer }); // Save bot's response
  } catch (e) {
    loader.textContent = 'Error: ' + e.message;
  }
}

// WebSocket setup for Chatwoot
const chatwootSocket = new WebSocket('wss://app.chatwoot.com/cable');

chatwootSocket.onopen = () => {
  console.log('Connected to Chatwoot WebSocket');
  // Replace 'conversation_id' with the actual conversation ID from the backend
  const conversationId = window.chatwootConversationId || null;
  if (conversationId) {
    chatwootSocket.send(JSON.stringify({
      command: 'subscribe',
      identifier: JSON.stringify({
        channel: 'RoomChannel',
        id: conversationId
      })
    }));
  }
};

chatwootSocket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('[DEBUG] WebSocket message received:', data); // Debugging print

  if (data.type === 'ping' || !data.message) return;

  const messageData = data.message.data;
  console.log('[DEBUG] Parsed message data:', messageData); // Debugging print

  if (messageData && messageData.message_type === 1) { // Check if it's an outgoing message (from agent)
    const message = messageData.content;
    const sender = messageData.sender.name || "Agent"; // Use sender's name or default to "Agent"
    console.log(`[DEBUG] Incoming message from ${sender}:`, message); // Debugging print
    append('bot', `${sender}: ${message}`); // Display the message in the chat box
  }
};

chatwootSocket.onclose = () => {
  console.log('Disconnected from Chatwoot WebSocket');
};

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
