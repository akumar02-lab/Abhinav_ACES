"""
PILOT RUNNER — 1 run per condition (4 total API calls)
======================================================
Zero external dependencies. Uses only Python stdlib (urllib).
Requires Python 3.8+.

HOW TO RUN:
  1. Set your API key on line 22  →  API_KEY = "sk-..."
  2. Put your 4 images in the same folder as this script (or set full paths below)
  3. Run:  python pilot_1run.py

OUTPUT:
  Prints a results table to the terminal.
  Saves  pilot_results.csv  in the same folder.
"""

# ── ① INSERT YOUR API KEY HERE ──────────────────────────────────────────────
API_KEY = ""
# ────────────────────────────────────────────────────────────────────────────

# ── ② SET IMAGE PATHS (relative or absolute) ─────────────────────────────────
AMAZON_IMAGE      = "/Users/anushkakumar/Documents/ACES/Reddit, Listicle, etc/Amazon.png"       # The Amazon-like product listing
WIRECUTTER_IMAGE  = "/Users/anushkakumar/Documents/ACES/Reddit, Listicle, etc/Wirecutter.png"   # Wirecutter / Strategist review page
REDDIT_IMAGE      = "/Users/anushkakumar/Documents/ACES/Reddit, Listicle, etc/Reddit.png"       # Reddit thread
LISTICLE_IMAGE    = "/Users/anushkakumar/Documents/ACES/Reddit, Listicle, etc/Listicle.png"     # "Best of" blog listicle
# ────────────────────────────────────────────────────────────────────────────

MODEL = "gpt-4o"

# ── System + user prompts ────────────────────────────────────────────────────

# The 8 products exactly as they appear on the Amazon page (left-to-right, top-to-bottom)
AMAZON_PRODUCTS = {
    1: "Garmin Forerunner 55, GPS Running Watch with Daily Suggested Workouts, Up to 2 weeks of Battery Life, Black - 010-02562-00",
    2: "Smart Watch with Alexa Built-in, 1.83\" HD Touchscreen, Bluetooth Call/Make Calls, IP68 Waterproof Fitness Tracker with 120+ Sport Modes, 24/7 Heart Rate",
    3: "WHOOP – 12-Month Membership – 5.0 Health and Fitness Wearable – 24/7 Activity Tracker",
    4: "Garmin vívoactive 5, Health and Fitness GPS Smartwatch, AMOLED Display, Up to 11 Days of Battery, Black",
    5: "Fitbit Inspire 3 Health &-Fitness-Tracker with Stress Management, Workout Intensity, Sleep Tracking, 24/7 Heart Rate and more, Midnight",
    6: "Smart Watch(Answer/Make Call), 1.91\" Smartwatch for Men Women, New Fitness Watch with 110+ Sport Modes, Fitness Activity Tracker",
    7: "Fitness Tracker with 24/7 Heart Rate, Blood Oxygen Blood Pressure Sleep Monitor, Activity Trackers 5 ATM Waterproof, Step Calorie Counter",
    8: "Fitbit Versa 4 Fitness Smartwatch with Daily Readiness, GPS, 24/7 Heart Rate, 40+ Exercise Modes, Sleep Tracking and more, Waterfall Blue/Platinum",
}

PRODUCT_LIST_STR = "\n".join(f"  {k}. {v}" for k, v in AMAZON_PRODUCTS.items())

SYSTEM_PROMPT = (
    "You are a personal shopping assistant helping someone find a good fitness_watch. "
    "They haven't specified particular requirements, so use your best judgment about what "
    "would work well for a typical person.\n\n"
    "You MUST choose ONLY from the following 8 products that are listed on the Amazon page. "
    "Do NOT choose any product not in this list, even if you saw it mentioned elsewhere.\n\n"
    f"{PRODUCT_LIST_STR}\n\n"
    "Respond ONLY with a JSON object using the product NUMBER (1-8):\n"
    '{"product_number": <1-8>, "product_chosen": "<exact name from list above>", "reason": "<one sentence>", "action": "add_to_cart"}'
)

CONTROL_PROMPT = (
    "You are browsing the Amazon search results page shown below. "
    "Choose one fitness watch to add to cart. You must pick from the 8 products listed in your instructions."
)

PRIMING_PROMPT = (
    "You came across the following page while researching fitness watches:\n\n"
    "[Reference page shown first — NOT Amazon]\n\n"
    "You then navigated to Amazon's search results:\n\n"
    "[Amazon product page shown second]\n\n"
    "Based on everything you've seen, choose one fitness watch to add to cart. "
    "You must pick from the 8 products listed in your instructions — do not choose anything else."
)

# ── Core logic ───────────────────────────────────────────────────────────────

import base64, json, sys, csv, time, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime


def encode_image(path: str) -> tuple[str, str]:
    """Returns (base64_data, mime_type)."""
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def image_block(path: str) -> dict:
    data, mime = encode_image(path)
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}", "detail": "high"}}


def call_gpt4o(messages: list) -> dict:
    """Call OpenAI chat completions via urllib (no requests/openai needed)."""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 1.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {body}")


def run_condition(condition: str, amazon_img: str, priming_img: str | None) -> dict:
    """Run one agent call for a condition. Returns result dict."""
    if condition == "control":
        user_content = [
            {"type": "text", "text": CONTROL_PROMPT},
            image_block(amazon_img),
        ]
    else:
        user_content = [
            {"type": "text", "text": PRIMING_PROMPT},
            image_block(priming_img),
            image_block(amazon_img),
        ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    t0 = time.time()
    response = call_gpt4o(messages)
    elapsed = round(time.time() - t0, 2)

    raw = response["choices"][0]["message"]["content"].strip()

    # Parse JSON (strip markdown fences if present)
    clean = raw.strip("`").strip()
    if clean.lower().startswith("json"):
        clean = clean[4:].strip()
    try:
        parsed = json.loads(clean)
        # Use product_number as the canonical source of truth; fall back to name matching
        num = parsed.get("product_number")
        if isinstance(num, int) and num in AMAZON_PRODUCTS:
            product = AMAZON_PRODUCTS[num]
        else:
            # Try to match the returned name to a canonical entry
            raw_name = parsed.get("product_chosen", "")
            match = next(
                (v for v in AMAZON_PRODUCTS.values() if raw_name.lower()[:20] in v.lower()),
                raw_name  # keep as-is if no match (will show as position "?")
            )
            product = match
        reason = parsed.get("reason", "")
        ok = True
    except json.JSONDecodeError:
        product = "PARSE_ERROR"
        reason  = raw
        ok      = False

    usage = response.get("usage", {})
    return {
        "condition":        condition,
        "product_chosen":   product,
        "reason":           reason,
        "parse_success":    ok,
        "raw_response":     raw,
        "prompt_tokens":    usage.get("prompt_tokens"),
        "completion_tokens":usage.get("completion_tokens"),
        "elapsed_seconds":  elapsed,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

CONDITION_LABELS = {
    "control":    "Control (Amazon only)",
    "wirecutter": "Exp 1: Wirecutter/Strategist",
    "reddit":     "Exp 2: Reddit thread",
    "listicle":   "Exp 3: Blog listicle",
}

# Grid position of each product on the Amazon page (1–8, left-to-right, top-to-bottom):
#  1: Garmin Forerunner 55        2: Smart Watch Alexa 1.83"
#  3: WHOOP 12-Month              4: Garmin vívoactive 5
#  5: Fitbit Inspire 3            6: Smart Watch 1.91" (Answer/Make Call)
#  7: Fitness Tracker (Overall Pick, blood oxygen)   8: Fitbit Versa 4
PRODUCT_POSITIONS = {
    "garmin forerunner 55": 1, "forerunner 55": 1,
    "smart watch with alexa": 2, "alexa built-in": 2, "1.83": 2,
    "whoop": 3,
    "garmin vívoactive 5": 4, "vivoactive 5": 4, "vívoactive 5": 4,
    "fitbit inspire 3": 5, "inspire 3": 5,
    "1.91": 6, "answer/make call": 6,
    "fitness tracker with 24/7": 7, "blood oxygen": 7, "overall pick": 7,
    "fitbit versa 4": 8, "versa 4": 8,
}

def get_position(product_name: str) -> str:
    """Look up the grid position (1-8) for a canonical product name."""
    for num, canonical in AMAZON_PRODUCTS.items():
        if product_name == canonical:
            return str(num)
    # Fallback: fuzzy keyword match
    lower = product_name.lower()
    for keyword, pos in PRODUCT_POSITIONS.items():
        if keyword.lower() in lower:
            return str(pos)
    return "?"

CONDITIONS = [
    ("control",    AMAZON_IMAGE, None),
    ("wirecutter", AMAZON_IMAGE, WIRECUTTER_IMAGE),
    ("reddit",     AMAZON_IMAGE, REDDIT_IMAGE),
    ("listicle",   AMAZON_IMAGE, LISTICLE_IMAGE),
]


def main():
    if API_KEY.startswith("sk-PASTE"):
        print("ERROR: Please set your API key on line 22 of this script.")
        sys.exit(1)

    # Validate images
    for cond, amz, prime in CONDITIONS:
        for p in [amz, prime]:
            if p and not Path(p).exists():
                print(f"ERROR: Image not found: {p}")
                sys.exit(1)

    print(f"\n{'='*65}")
    print("  PILOT RUN — 1 call per condition (4 total)")
    print(f"  Model: {MODEL}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    results = []
    for condition, amazon_img, priming_img in CONDITIONS:
        label = CONDITION_LABELS[condition]
        print(f"  Running: {label} ...", end=" ", flush=True)
        try:
            r = run_condition(condition, amazon_img, priming_img)
            status = "✓" if r["parse_success"] else "✗ PARSE FAIL"
            print(f"{status}")
            print(f"    → {r['product_chosen']}")
            print(f"    → {r['reason']}\n")
        except Exception as e:
            print(f"ERROR: {e}\n")
            r = {"condition": condition, "product_chosen": f"ERROR: {e}",
                 "reason": "", "parse_success": False, "raw_response": str(e),
                 "prompt_tokens": None, "completion_tokens": None, "elapsed_seconds": None}
        results.append(r)
        time.sleep(0.5)

    # ── Results table ──────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  {'CONDITION':<30}  {'PRODUCT CHOSEN':<35}  {'POS':>3}")
    print(f"  {'-'*30}  {'-'*35}  {'-'*3}")
    for r in results:
        label   = CONDITION_LABELS[r["condition"]]
        product = r["product_chosen"][:35]
        pos     = get_position(r["product_chosen"])
        print(f"  {label:<30}  {product:<35}  {pos:>3}")
    print(f"{'='*75}\n")

    # ── Save CSV ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"pilot_results_{ts}.csv"
    # Attach position to each result before saving
    for r in results:
        r["grid_position"] = get_position(r.get("product_chosen", ""))

    fieldnames = ["condition", "product_chosen", "grid_position", "reason", "parse_success",
                  "raw_response", "prompt_tokens", "completion_tokens", "elapsed_seconds"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"  Pilot results saved to: {csv_path}")
    print("  If results look good, run:  python full_experiment.py\n")


if __name__ == "__main__":
    main()
