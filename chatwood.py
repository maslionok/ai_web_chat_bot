import requests

# === 🔧 CONFIG ===
BASE_URL = "https://app.chatwoot.com"  # or your self-hosted URL
ACCOUNT_ID = "120028"
INBOX_ID = "64217"
API_TOKEN = "1L242MucDr7mnyLvPA3SgywJ"

HEADERS = {
    "Content-Type": "application/json",
    "api_access_token": API_TOKEN
}
random_number = 1234567890  # Replace with a random number generator if needed
# === 1. Create a contact ===
contact_payload = {
    "name": "Test Client",
    "email": f"testclient{random_number}@example.com",
    "inbox_id": INBOX_ID
}
r = requests.post(
    f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/contacts",
    headers=HEADERS,
    json=contact_payload
)
contact_data = r.json()
print("Contact creation response:", r.status_code, r.text)

contact_id = contact_data["payload"]["contact"]["id"]
source_id = contact_data["payload"]["contact"]["contact_inboxes"][0]["source_id"]

print("✅ Contact created")

# === 2. Create a conversation ===
conversation_payload = {
    "source_id": source_id,
    "inbox_id": INBOX_ID,
    "contact_id": contact_id
}
r = requests.post(
    f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations",
    headers=HEADERS,
    json=conversation_payload
)
conversation_id = r.json()["id"]

print("✅ Conversation created")

# === 3. Send a message as the contact ===
message_payload = {
    "content": "Hi, this is a test from the API!",
    "message_type": "incoming"
}
r = requests.post(
    f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}/messages",
    headers=HEADERS,
    json=message_payload
)

if r.status_code == 200:
    print("✅ Message sent successfully!")
else:
    print("❌ Failed to send message:", r.status_code, r.text)
