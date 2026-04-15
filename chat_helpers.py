import tiktoken

MAX_TOKENS = 16000

def get_db_sys_prompt(warmth=True, personality_dict={}):

    with open(f"SYS_PROMPT.txt", "r") as f:
        sys_prompt = f.read()
    
    # --- WARMTH MANIPULATION ---
    if warmth:
        warmth_prompt = """
        Warmth is ON for this conversation.
        - In the {OPENING_LINE} slot: use exactly one of these two greetings (pick whichever fits naturally):
          Option A: "Hi, good question 😊🌊"
          Option B: "Hey, amazing question 😊🌊"
          Do NOT invent other greetings. Use one of the two above verbatim.
        - If personalization is also ON, the personalized framing sentence goes on a NEW LINE (second line) directly after the greeting. The personalized sentence must NOT contain any emojis.
        - In the {EXPLANATION_PARAGRAPH} slot: append exactly one emoji at the end of the paragraph.
        - In the {CLOSING_SENTENCE} slot: append exactly one emoji at the end of the sentence.
        - In the {PERSONALIZED_CLOSING_SENTENCE} slot (only if personalization is also ON): append exactly one emoji at the end.
        - Do NOT place emojis anywhere else in the response (not in bullets, not in headers).
        - Do NOT change the structure, number of sections, or number of bullet points because of warmth.
        """
    else:
        warmth_prompt = """
        Warmth is OFF for this conversation.
        - Do NOT use any emojis anywhere.
        - Do NOT use warm greetings, praise, or friendly closings.
        - Start with a direct, neutral opening sentence (one line only).
        - End with a direct, neutral summary sentence.
        """

    warmth_prompt = f"<warmth>\n{warmth_prompt}\n</warmth>"
    sys_prompt = sys_prompt.replace("<warmth>", warmth_prompt)

    # --- PERSONALIZATION MANIPULATION ---
    if personality_dict:
        personalization = f"""
        Personalization is ON for this conversation.

        Here is information about the user:
        {personality_dict}

        Personalization rules — follow ALL of these:
        - In the {{OPENING_LINE}} slot:
          - If warmth is ON: the warm greeting is on line 1. Add a personalized framing sentence on line 2 (new line). Line 2 must NOT contain emojis.
          - If warmth is OFF: the opening is ONE single sentence that includes personalized framing referencing the user's background (e.g., "Given your background in X, think of this like..."). No emojis. No second line.
        - In the {{EXPLANATION_PARAGRAPH}} (Section 1 body): include exactly ONE analogy or example drawn from the user's background to explain the risk concept. Use terms familiar to the user's field.
        - Do NOT add personalization references in the bullet points (Section 2).
        - In the {{PERSONALIZED_CLOSING_SENTENCE}} slot: add one sentence after the Bottom line's {{CLOSING_SENTENCE}} that ties the conclusion back to the user's background (e.g., "From your experience in X, you'd recognize this as..."). If warmth is ON, append one emoji. If warmth is OFF, no emoji.
        - Include one phrase like "given your background" or "based on your background" to make personalization explicit. Place it in either the opening personalization line or the personalized closing sentence — not both.
        - Personalization fills specific slots in the template. Do NOT restructure the response to accommodate personalization.
        - Do NOT change the number of sections, bullets, or overall structure because of personalization.
        """
    else:
        personalization = """
        Personalization is OFF for this conversation.
        - Do NOT reference any user background, hobbies, or field of study.
        - Do NOT use analogies from specific disciplines.
        - Do NOT use phrases like "given your background" or "based on your background."
        - Do NOT include a {PERSONALIZED_CLOSING_SENTENCE}. Omit it entirely.
        - Explain everything in plain, general terms.
        """

    personalization = f"<personalization>\n{personalization}\n</personalization>"
    sys_prompt = sys_prompt.replace("<personalization>", personalization)

    return sys_prompt

def build_input_from_history(message, history, warmth=True, personality_dict={}):

    parts = []
    parts.append({"role": "system", "content": get_db_sys_prompt(warmth=warmth, personality_dict=personality_dict)})
    
    for msg in history:
        if msg["role"] == "user":
            parts.append({"role": "user", "content": msg["content"]})
        if msg["role"] == "assistant":
            parts.append({"role": "assistant", "content": msg["content"]})
    parts.append({"role": "user", "content": message})
    parts = truncate_history(parts, MAX_TOKENS)

    return parts

def count_tokens(messages, model="gpt-5.4"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for msg in messages:
        num_tokens += len(enc.encode(msg["content"]))
    return num_tokens

def truncate_history(messages, max_tokens=MAX_TOKENS, model="gpt-5.4"):
    while count_tokens(messages, model=model) > max_tokens and len(messages) > 2:
        messages.pop(1)
    return messages