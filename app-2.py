import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hotel-assistant")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. LOAD KNOWLEDGE BASE FROM JSON
# ---------------------------------------------------------------------------
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
with open(KB_PATH, "r") as f:
    KB = json.load(f)

# Convert booked_dates lists to sets for O(1) lookup
for room in KB["rooms"]:
    room["booked_dates"] = set(room["booked_dates"])

HOTEL_INFO_TEXT = f"""
Hotel: {KB['hotel_name']}
Check-in time: {KB['check_in_time']}
Check-out time: {KB['check_out_time']}
Swimming pool: {KB['amenities']['swimming_pool']}
Breakfast: {KB['amenities']['breakfast']}
WiFi: {KB['amenities']['wifi']}
Gym: {KB['amenities']['gym']}
Restaurant: {KB['amenities']['restaurant']}
Airport shuttle: {KB['amenities']['airport_shuttle']}
Cancellation policy: {KB['cancellation_policy']}
"""

# ---------------------------------------------------------------------------
# 2. DETERMINISTIC BUSINESS LOGIC (kept OUTSIDE the LLM, as required)
# ---------------------------------------------------------------------------

def _date_range(check_in: str, check_out: str):
    """Return the list of nights (as date strings) between check_in and check_out (exclusive of checkout day)."""
    d1 = datetime.strptime(check_in, "%Y-%m-%d")
    d2 = datetime.strptime(check_out, "%Y-%m-%d")
    nights = []
    cur = d1
    while cur < d2:
        nights.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return nights


def check_availability(check_in: str, check_out: str, adults: int) -> dict:
    """Check which room types are available for a full date range and guest count.

    Args:
        check_in: Check-in date in YYYY-MM-DD format.
        check_out: Check-out date in YYYY-MM-DD format.
        adults: Number of adult guests who need to be accommodated.
    """
    try:
        nights = _date_range(check_in, check_out)
    except ValueError:
        return {"error": "Dates must be in YYYY-MM-DD format."}

    if not nights:
        return {"error": "Check-out date must be after check-in date."}

    available = []
    for room in KB["rooms"]:
        if room["capacity"] < adults:
            continue
        if any(night in room["booked_dates"] for night in nights):
            continue
        available.append(
            {
                "name": room["name"],
                "capacity": room["capacity"],
                "price_per_night": room["price_per_night"],
                "total_price": room["price_per_night"] * len(nights),
                "nights": len(nights),
            }
        )

    if not available:
        return {
            "check_in": check_in,
            "check_out": check_out,
            "adults": adults,
            "available_rooms": [],
            "message": "No rooms available for these dates and guest count.",
        }

    return {
        "check_in": check_in,
        "check_out": check_out,
        "adults": adults,
        "available_rooms": available,
    }


TOOLS = [check_availability]

# ---------------------------------------------------------------------------
# 3. GEMINI CLIENT + CONVERSATION CONTEXT (simple in-memory store)
# ---------------------------------------------------------------------------

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# conversation_id -> list of {"role": "user"/"model", "text": str}
CONVERSATIONS: dict[str, list] = {}

SYSTEM_PROMPT = f"""You are a friendly guest assistant for {KB['hotel_name']}.
Answer questions ONLY using the hotel information provided below, and the
check_availability tool for date/guest-based room queries. Keep replies
concise and warm.

If a guest asks something you cannot answer from the information given (e.g.
something not covered by hotel info or availability data), say so honestly
and suggest they contact the hotel directly instead of guessing or inventing
details.

Today's date is {datetime.now().strftime('%Y-%m-%d')}.

HOTEL INFO:
{HOTEL_INFO_TEXT}
"""


def get_gemini_reply(conversation_id: str, user_message: str) -> str:
    history = CONVERSATIONS.setdefault(conversation_id, [])
    contents = []
    for turn in history:
        contents.append(types.Content(role=turn["role"], parts=[types.Part(text=turn["text"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=TOOLS,
        ),
    )

    reply_text = response.text or "Sorry, I couldn't generate a response. Please try rephrasing."

    history.append({"role": "user", "text": user_message})
    history.append({"role": "model", "text": reply_text})
    # Keep context bounded so it doesn't grow unbounded in memory
    CONVERSATIONS[conversation_id] = history[-20:]

    return reply_text


# ---------------------------------------------------------------------------
# 4. API ENDPOINTS
# ---------------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    conversation_id = data.get("conversation_id") or str(uuid.uuid4())

    if not user_message:
        return jsonify({"error": "Message cannot be empty.", "conversation_id": conversation_id}), 400
    if len(user_message) > 1000:
        return jsonify({"error": "Message is too long (max 1000 characters).", "conversation_id": conversation_id}), 400

    logger.info(f"[{conversation_id}] User: {user_message}")

    try:
        reply = get_gemini_reply(conversation_id, user_message)
        logger.info(f"[{conversation_id}] Assistant: {reply[:200]}")
        return jsonify({"reply": reply, "conversation_id": conversation_id})
    except Exception as e:
        logger.exception(f"[{conversation_id}] Error handling chat request")
        return jsonify({
            "error": "The assistant is temporarily unavailable. Please try again in a moment.",
            "conversation_id": conversation_id,
        }), 500


@app.route("/availability", methods=["POST"])
def availability():
    """Direct structured endpoint for the frontend's date/guest picker,
    bypassing the LLM entirely for a fast, deterministic result."""
    data = request.get_json(silent=True) or {}
    check_in = data.get("check_in")
    check_out = data.get("check_out")
    adults = data.get("adults")

    if not check_in or not check_out or not adults:
        return jsonify({"error": "check_in, check_out and adults are all required."}), 400
    try:
        adults = int(adults)
        if adults < 1:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "adults must be a positive integer."}), 400

    result = check_availability(check_in, check_out, adults)
    if "error" in result:
        return jsonify(result), 400

    logger.info(f"Availability check: {check_in} to {check_out} for {adults} adults -> {len(result['available_rooms'])} rooms")
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# 5. FRONTEND
# ---------------------------------------------------------------------------

PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Sunrise Grand — Guest Assistant</title>
<style>
  body { font-family: -apple-system, Arial, sans-serif; background:#f4f1ec; margin:0; padding:0; }
  .wrap { max-width: 480px; margin: 0 auto; min-height: 100vh; display:flex; flex-direction:column; }
  header { background:#8c5a3c; color:#fff; padding:16px; text-align:center; font-weight:600; }
  #log { flex:1; overflow-y:auto; padding:16px; }
  .msg { margin:8px 0; padding:10px 14px; border-radius:14px; max-width:80%; line-height:1.4; white-space:pre-wrap; }
  .user { background:#8c5a3c; color:#fff; margin-left:auto; border-bottom-right-radius:4px; }
  .bot { background:#fff; border:1px solid #e0dcd4; margin-right:auto; border-bottom-left-radius:4px; }
  .error { background:#fdecea; color:#a83232; border:1px solid #f3c2c2; margin-right:auto; }
  form { display:flex; padding:12px; border-top:1px solid #e0dcd4; background:#fff; }
  input[type=text] { flex:1; padding:10px 12px; border-radius:20px; border:1px solid #ccc; outline:none; }
  button { margin-left:8px; padding:10px 16px; border:none; border-radius:20px; background:#8c5a3c; color:#fff; }
  button:disabled { background:#c7b3a5; }
  .avail-toggle { padding: 8px 16px; background:#fff; border-bottom:1px solid #e0dcd4; font-size:14px; }
  .avail-toggle a { color:#8c5a3c; }
  #availForm { display:none; padding:12px 16px; background:#fff; border-bottom:1px solid #e0dcd4; }
  #availForm label { display:block; font-size:12px; color:#555; margin-top:8px; }
  #availForm input { width:100%; padding:8px; box-sizing:border-box; border-radius:8px; border:1px solid #ccc; margin-top:2px; }
  #availForm button { width:100%; margin:12px 0 0 0; padding:10px; }
</style>
</head>
<body>
<div class="wrap">
  <header>Sunrise Grand — Guest Assistant</header>
  <div class="avail-toggle"><a href="#" id="toggleAvail">Check room availability for specific dates →</a></div>
  <div id="availForm">
    <label>Check-in date</label>
    <input type="date" id="checkIn" />
    <label>Check-out date</label>
    <input type="date" id="checkOut" />
    <label>Number of guests</label>
    <input type="number" id="adults" min="1" value="2" />
    <button id="availSubmit">Check availability</button>
  </div>
  <div id="log"></div>
  <form id="f">
    <input type="text" id="input" placeholder="Ask about check-in, breakfast, policies..." autocomplete="off" />
    <button type="submit" id="sendBtn">Send</button>
  </form>
</div>
<script>
const log = document.getElementById('log');
const form = document.getElementById('f');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
let conversationId = null;

function addMsg(text, cls) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

addMsg("Hi! I'm the Sunrise Grand guest assistant. Ask me about check-in, the pool, breakfast, or cancellation policy — or use the availability checker above for specific dates.", "bot");

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addMsg(text, 'user');
  input.value = '';
  sendBtn.disabled = true;
  const placeholder = addMsg('Thinking...', 'bot');
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, conversation_id: conversationId})
    });
    const data = await res.json();
    if (!res.ok) {
      placeholder.className = 'msg error';
      placeholder.textContent = data.error || 'Something went wrong. Please try again.';
    } else {
      conversationId = data.conversation_id;
      placeholder.textContent = data.reply;
    }
  } catch (err) {
    placeholder.className = 'msg error';
    placeholder.textContent = 'Network error — please check your connection and try again.';
  } finally {
    sendBtn.disabled = false;
  }
});

document.getElementById('toggleAvail').addEventListener('click', (e) => {
  e.preventDefault();
  const f = document.getElementById('availForm');
  f.style.display = f.style.display === 'none' ? 'block' : 'none';
});

document.getElementById('availSubmit').addEventListener('click', async () => {
  const checkIn = document.getElementById('checkIn').value;
  const checkOut = document.getElementById('checkOut').value;
  const adults = document.getElementById('adults').value;
  if (!checkIn || !checkOut || !adults) {
    addMsg('Please fill in check-in date, check-out date, and number of guests.', 'error');
    return;
  }
  addMsg(`Checking availability: ${checkIn} to ${checkOut}, ${adults} guest(s)`, 'user');
  const placeholder = addMsg('Checking...', 'bot');
  try {
    const res = await fetch('/availability', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({check_in: checkIn, check_out: checkOut, adults: adults})
    });
    const data = await res.json();
    if (!res.ok) {
      placeholder.className = 'msg error';
      placeholder.textContent = data.error || 'Could not check availability.';
      return;
    }
    if (!data.available_rooms || data.available_rooms.length === 0) {
      placeholder.textContent = 'No rooms available for those dates and guest count.';
    } else {
      let text = 'Available rooms:\\n';
      data.available_rooms.forEach(r => {
        text += `• ${r.name} (fits ${r.capacity}) — ₹${r.price_per_night}/night, ₹${r.total_price} total for ${r.nights} night(s)\\n`;
      });
      placeholder.textContent = text.trim();
    }
  } catch (err) {
    placeholder.className = 'msg error';
    placeholder.textContent = 'Network error — please try again.';
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
