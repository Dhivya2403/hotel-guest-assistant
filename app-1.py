import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from google import genai
from google.genai import types

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. MOCK HOTEL DATA (edit these to make it your own / more realistic)
# ---------------------------------------------------------------------------

HOTEL_INFO = """
Hotel: Sunrise Grand
Check-in time: 2:00 PM
Check-out time: 11:00 AM
Swimming pool: Yes, open 6 AM - 9 PM, rooftop infinity pool
Breakfast: Complimentary buffet breakfast included with every booking, served 7 AM - 10:30 AM
Cancellation policy: Free cancellation up to 48 hours before check-in. Cancellations within 48 hours are charged one night's fare.
Other amenities: Free WiFi, gym, in-house restaurant, airport shuttle on request.
"""

# Each room type: id, name, capacity (max guests), price/night, and a set of
# dates (YYYY-MM-DD) that are ALREADY BOOKED (i.e. unavailable) for that room.
ROOMS = [
    {
        "id": "standard",
        "name": "Standard Room",
        "capacity": 2,
        "price_per_night": 3500,
        "booked_dates": {"2026-09-20", "2026-09-21"},
    },
    {
        "id": "deluxe",
        "name": "Deluxe Room",
        "capacity": 3,
        "price_per_night": 5000,
        "booked_dates": {"2026-09-25"},
    },
    {
        "id": "family_suite",
        "name": "Family Suite",
        "capacity": 4,
        "price_per_night": 7500,
        "booked_dates": set(),
    },
]


# ---------------------------------------------------------------------------
# 2. "TOOL" FUNCTIONS — real Python logic, not left to the LLM to guess.
#    The Gemini SDK reads each function's type hints + docstring to know
#    when and how to call it (automatic function calling).
# ---------------------------------------------------------------------------

def check_availability(date: str) -> dict:
    """Check which hotel room types are available on a specific date.

    Args:
        date: The date to check, in YYYY-MM-DD format.
    """
    available = []
    for room in ROOMS:
        if date not in room["booked_dates"]:
            available.append(
                {"name": room["name"], "capacity": room["capacity"], "price_per_night": room["price_per_night"]}
            )
    return {"date": date, "available_rooms": available}


def suggest_room(guests: int) -> dict:
    """Suggest the best (cheapest fitting) room type for a given number of guests.

    Args:
        guests: The number of guests who need to stay in the room.
    """
    fitting = [r for r in ROOMS if r["capacity"] >= guests]
    if not fitting:
        return {"message": f"No single room accommodates {guests} guests. Consider booking two rooms."}
    best = min(fitting, key=lambda r: r["price_per_night"])
    return {
        "recommended_room": best["name"],
        "capacity": best["capacity"],
        "price_per_night": best["price_per_night"],
    }


# ---------------------------------------------------------------------------
# 3. GEMINI CLIENT
# ---------------------------------------------------------------------------

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = f"""You are a friendly hotel guest assistant for Sunrise Grand.
Answer guest questions using the hotel information below. For questions about
room availability on a specific date, or which room suits a number of guests,
use the provided tools instead of guessing. Keep answers concise and warm.
Today's date is {datetime.now().strftime('%Y-%m-%d')}.

HOTEL INFO:
{HOTEL_INFO}
"""


# ---------------------------------------------------------------------------
# 4. CHAT ENDPOINT
# ---------------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[check_availability, suggest_room],
            ),
        )
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"reply": f"Sorry, something went wrong: {e}"}), 500


# ---------------------------------------------------------------------------
# 5. FRONTEND — single page, no build step needed
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
  .wrap { max-width: 480px; margin: 0 auto; height: 100vh; display:flex; flex-direction:column; }
  header { background:#8c5a3c; color:#fff; padding:16px; text-align:center; font-weight:600; }
  #log { flex:1; overflow-y:auto; padding:16px; }
  .msg { margin:8px 0; padding:10px 14px; border-radius:14px; max-width:80%; line-height:1.4; }
  .user { background:#8c5a3c; color:#fff; margin-left:auto; border-bottom-right-radius:4px; }
  .bot { background:#fff; border:1px solid #e0dcd4; margin-right:auto; border-bottom-left-radius:4px; }
  form { display:flex; padding:12px; border-top:1px solid #e0dcd4; background:#fff; }
  input { flex:1; padding:10px 12px; border-radius:20px; border:1px solid #ccc; outline:none; }
  button { margin-left:8px; padding:10px 16px; border:none; border-radius:20px; background:#8c5a3c; color:#fff; }
</style>
</head>
<body>
<div class="wrap">
  <header>Sunrise Grand — Guest Assistant</header>
  <div id="log"></div>
  <form id="f">
    <input id="input" placeholder="Ask about check-in, rooms, dates..." autocomplete="off" />
    <button type="submit">Send</button>
  </form>
</div>
<script>
const log = document.getElementById('log');
const form = document.getElementById('f');
const input = document.getElementById('input');

function addMsg(text, cls) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

addMsg("Hi! I'm the Sunrise Grand guest assistant. Ask me about check-in, the pool, breakfast, cancellation policy, or room availability.", "bot");

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addMsg(text, 'user');
  input.value = '';
  addMsg('...', 'bot');
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await res.json();
    log.lastChild.textContent = data.reply;
  } catch (err) {
    log.lastChild.textContent = 'Something went wrong. Please try again.';
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
