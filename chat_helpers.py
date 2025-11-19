import tiktoken

MAX_TOKENS = 16000

def get_db_sys_prompt(competence=True, personality_dict={}):

    with open(f"SYS_PROMPT.txt", "r") as f:
        sys_prompt = f.read()
    
    if competence:
        chat_style = """When answering, you must:

- Sound assertive, precise, and knowledgeable.
- Use a clear and professional language while being friendly and helpful. 
- Avoid hedging words (e.g., “maybe”, “I think”, “possibly”).
- Do not introduce the possibility of being wrong.
- Answer as your answer is correct and you are 100% sure about it."""

    else:
        chat_style = """When answering, you must:

- Remind the user that it is impossible to estimate the risks with absolute certainty.
- Use a clear and professional language while not being assertive or too confident. 
- Use hedging words (e.g., “maybe”, “I think”, “possibly”) that introduce uncertainty.
- Avoid confident words like “definitely”, “clearly”, or “certainly”.
- Remind the user that your answer might be incorrect."""
    
    sys_prompt = sys_prompt.replace("<chat_style>", chat_style)

    if personality_dict:
        personalization = f"""Here is some information about the user:
        {personality_dict}
        Use this information to explain the topic in a personalized way.
        If a nickname is provided, use it to address the user. If it is N/A or not provided, do not address the user.
        When explaining the topic, give personalized examples based on the user's background.
        Help the user understand the topic by tailoring the explanation to their background.
        Use analogies based on the user's background.
        Keep the user's age in mind when explaining the topic.
        Make sure the answer is thoroughly personalized.
        Make sure the personalized response makes sense and incentivizes user to accept your point of view.
        Don't mention that you are deliberately personalizing the answer, keep it hidden from the user.
        """
    else:
        personalization = ""
    sys_prompt = sys_prompt.replace("<personalization>", personalization)

    return sys_prompt
    
def build_input_from_history(message, history, competence=True, personality_dict={}):

    parts = []
    parts.append({"role": "system", "content": get_db_sys_prompt(competence=competence, personality_dict=personality_dict)})
    
    for msg in history:
        if msg["role"] == "user":
            parts.append({"role": "user", "content": msg["content"]})
        if msg["role"] == "assistant":
            parts.append({"role": "assistant", "content": msg["content"]})
    parts.append({"role": "user", "content": message})
    parts = truncate_history(parts, MAX_TOKENS)

    return parts

def count_tokens(messages, model="gpt-4.1"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for msg in messages:
        num_tokens += len(enc.encode(msg["content"]))
    return num_tokens

def truncate_history(messages, max_tokens=MAX_TOKENS, model="gpt-4.1"):
    while count_tokens(messages, model=model) > max_tokens and len(messages) > 2:
        messages.pop(1)
    return messages
