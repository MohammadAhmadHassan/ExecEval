"""
STEP 1: Connectivity test. Confirms each API key works with ONE tiny call
before we do anything at scale. Run this FIRST.

Setup (one time):
    pip install openai anthropic google-genai python-dotenv
    - copy .env.example to .env and paste your three real keys into it

Run:
    python test_connectivity.py
It asks each model to translate a trivial Python line to Java and prints
the result (or a clear error) for each. No CodeHalu data touched.
"""

import os
from dotenv import load_dotenv
load_dotenv()

TINY_PY = "print(sum(map(int, input().split())))"
PROMPT = ("Translate this Python program to Java. It reads from stdin and "
          "writes to stdout. Output ONLY the Java code, no explanation, no "
          "markdown fences. Use a public class named Main.\n\n" + TINY_PY)

def test_openai():
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.chat.completions.create(
        model="gpt-4o-mini",                # cheap, reliable; change later if you like
        messages=[{"role":"user","content":PROMPT}],
        max_tokens=400, temperature=0)
    return r.choices[0].message.content

def test_anthropic():
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = client.messages.create(
        model="claude-3-5-haiku-20241022",  # cheap tier; change later if you like
        max_tokens=400, temperature=0,
        messages=[{"role":"user","content":PROMPT}])
    return r.content[0].text

def test_gemini():
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = client.models.generate_content(
        model="gemini-2.0-flash",           # free-tier friendly
        contents=PROMPT)
    return r.text

for name, fn in [("OpenAI (gpt-4o-mini)", test_openai),
                 ("Anthropic (claude-3.5-haiku)", test_anthropic),
                 ("Gemini (2.0-flash)", test_gemini)]:
    print("="*60)
    print(name)
    print("="*60)
    try:
        out = fn()
        print("OK -- got a response:\n")
        print(out.strip()[:500])
    except KeyError as e:
        print(f"FAIL: missing key in .env -> {e}")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
    print()
