import tiktoken

MAX_TOKENS = 16000

def get_db_sys_prompt(warmth=True, personality_dict={}):

    with open(f"SYS_PROMPT.txt", "r") as f:
        sys_prompt = f.read()
    
    # --- WARMTH MANIPULATION ---
    if warmth:
        warmth_prompt = """
        Warmth is ON for this conversation.
        - In the {OPENING_LINE} slot: use exactly one of these three greetings (pick whichever fits naturally):
          Option A: "Great question 😄💛"
          Option B: "Amazing question 🤗💛"
          Option C: "Good question! 😊💛"
          Do NOT invent other greetings. Use one of the three above verbatim.
        - The response must start with this warm greeting line.
        - Do NOT add a personalized sentence inside {OPENING_LINE}.
        - When emojis are allowed by the structure, use warm, friendly emojis (e.g., 💛, 🤗, 😄, 🌟, ✨, 💪, 🙌). Outside the fixed greeting options, avoid basic smiley faces like 🙂 and prefer hearts, sparkles, hugs, and enthusiastic expressions.
        - In the explanation sentence (the line before the "Why" header): append exactly one warm emoji at the end.
        - In the third (last) bullet point under "Why": append exactly one warm emoji at the end.
        - Do NOT place emojis anywhere else in the response (not in headers, not in the personalization bridge sentence, and not in the final budget-conclusion sentence).
        - Begin the final budget-conclusion sentence with a concluding transition phrase (e.g., "In conclusion,", "Therefore,", "Overall,", "So,").
        - Keep the final budget-conclusion sentence to one concise sentence only, without extra explanation.
        - Do NOT change the structure or number of bullet points because of warmth.
        """
    else:
        warmth_prompt = """
        Warmth is OFF for this conversation.
        - Do NOT use any emojis anywhere.
        - Do NOT use warm greetings, praise, or friendly closings.
        - Do NOT add a greeting line.
        - Start directly with the explanation sentence.
        - Put the budget-judgment sentence at the end (last line), with dynamic wording that clearly says the expert-recommended budget seems too high.
        - Begin the budget-judgment sentence with a concluding transition phrase (e.g., "In conclusion,", "Therefore,", "Overall,", "So,").
        - Keep the budget-judgment sentence to one concise sentence only, without extra explanation.
        - Keep a direct, neutral tone throughout.
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
        - Include one personalized bridge sentence between the explanation sentence and the "Why" header.
        - That personalized bridge sentence must explicitly name one real background item from the profile, e.g., "Let me explain it based on your background in chemistry."
        - Do NOT use a fixed generic sentence; the background phrase must adapt to the actual user profile values.
        - Choose the background for this personalized bridge sentence from available demographics that are relevant to the current explanation; if several fit, randomize among those.
        - In the single "Why" section, include exactly THREE personalized explanations in the bullet points (all three bullets personalized).
        - Make personalization explicit in all three bullets using clear markers such as "Given your background in X..." or "From your X workflow...".
        - Bullet composition rule: exactly ONE bullet should use a strong, realistic analogy ("Think of it as..."), while the other TWO should be direct, no-analogy explanations that use concrete terms/concepts from the user's background.
        - Choose the background references from profile items that are genuinely relevant to the specific risk mechanism; if multiple are relevant, randomize among those relevant options.
        - If multiple demographics are available (work, education, hobbies), distribute different demographics across the three personalized bullets when possible.
        - Personalization style should mirror the existing personalized framing style: concrete, domain-specific, and decision-relevant.
        - Use specific terms from the provided user profile (role, domain, workflow, tool, and interests). Avoid generic personalization like "in your field" without concrete details.
        - The personalized arguments must be decision-relevant and persuasive, not decorative. Show why the analogy supports claiming the expert budget is too high.
        - Keep the core mechanism faithful to the knowledge base first, then add the personalized analogy; personalization must not replace or distort the base argument.
        - Avoid speculative or technically dubious analogies. Do NOT claim unsupported capabilities (e.g., automatic self-correction) unless explicitly grounded in the knowledge base or user profile.
        - If the profile is weakly related, still personalize explicitly but rely on transferable process terms (monitoring, thresholds, iteration, reliability, fallback, maintenance) rather than forced domain analogies.
        - If no strong analogy exists, use a conservative analogy in exactly one bullet and keep the other two bullets strictly mechanism-first with background terminology.
        - Do NOT use vague placeholders such as "similar to your work" or "as you know"; be explicit and concrete with domain mechanisms.
        - Personalization fills specific slots in the template. Do NOT restructure the response to accommodate personalization.
        - Do NOT change the number of bullets or overall structure because of personalization.
        """
    else:
        personalization = """
        Personalization is OFF for this conversation.
        - Do NOT reference any user background, hobbies, or field of study.
        - Do NOT use analogies from specific disciplines.
        - Do NOT include the personalized bridge sentence.
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
