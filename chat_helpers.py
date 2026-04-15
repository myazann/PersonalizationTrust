import tiktoken

MAX_TOKENS = 16000

def get_db_sys_prompt(warmth=True, personality_dict={}):

    with open(f"SYS_PROMPT.txt", "r") as f:
        sys_prompt = f.read()
    
    if warmth:
        chat_style = """
        Use multiple emojis throughout your answer.
        Be very warm throughout your answer.
        Always start your answer with a friendly and very warm greeting or a warm praise of the question.
        """
    else:
        chat_style = ""

    # Have a positive and very enthusiastic tone.
    # Always end your answer with a friendly and very warm closing.

    chat_style = f"<chat_style>\n{chat_style}\n</chat_style>"
    sys_prompt = sys_prompt.replace("<chat_style>", chat_style)

    if personality_dict:
        personalization = f"""
        Here is some information about the user:
        {personality_dict}
        Your task is to provide user a personalized response given their background. 
        You should answer like you are someone who knows the user's background well. 

        Follow those rules:
        - The user's background is defined by the user's work/study and hobbies.
        - Use every aspect of the user's background in the response, meaning both their work/study and hobbies.
        - When explaining the topic, give personalized examples based on the user's background.
        - Explain the topic with terms from the user's background.
        - Include phrases like "given your background" or "based on your background" that makes personalization explicit.
        - Make sure to contextualize your answer based on the user's background. Contextualization is a personalization tactic that increases attention, interest, and motivation to process information by framing messages in a context that is meaningful to the recipient."""
    else:
        personalization = ""

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