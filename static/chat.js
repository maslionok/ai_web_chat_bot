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

// Append a message; returns the div
function append(who, text) {
  const div = document.createElement('div');
  div.className   = 'message ' + who;
  div.textContent = text;
  div.style.opacity = 0;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  setTimeout(() => { div.style.transition = 'opacity 0.25s'; div.style.opacity = 1; }, 10);
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
    question: text
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
  } catch (e) {
    loader.textContent = 'Error: ' + e.message;
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
