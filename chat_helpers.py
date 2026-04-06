import tiktoken

MAX_TOKENS = 16000

def get_db_sys_prompt(certainty=True, personality_dict={}):

    with open(f"SYS_PROMPT.txt", "r") as f:
        sys_prompt = f.read()
    
    if certainty:
        chat_style = """When answering, you must:
- Avoid hedging words (e.g., “maybe”, “possibly”).
- Use confident words like "definitely", "clearly", or "certainly"."""

    else:
        chat_style = """When answering, you must:
- Use hedging words (e.g., “maybe”, “possibly”).
- Avoid confident words like “definitely”, “clearly”, or “certainly”."""
    
    chat_style = f"<chat_style>\n{chat_style}\n</chat_style>"
    sys_prompt = sys_prompt.replace("<chat_style>", chat_style)

    if personality_dict:
        personalization = f"""Here is some information about the user:
{personality_dict}
Your task is to provide user a personalized response given their background. 
However, you have to keep this discreet and you should not make it explicit.
The user should not understand that you are deliberately personalizing the answer.
It should feel natural. You should answer like you are someone who knows the user's background well. 

Follow those rules:
- The user's background is defined by the user's work/study and hobbies.
- If a nickname or alias is provided, use it to address the user in each if your responses. If it is N/A or not provided, do not address the user. 
- When explaining the topic, give personalized examples based on the user's background.
- Explain the topic with terms from the user's background.
- Use every aspect of the user's background in the response, meaning both their work/study and hobbies.
- Don't mention that you are deliberately personalizing the answer, keep it hidden from the user. 
- Don't mention that you are tailoring your response to the user's background.
- Don't include phrases like "given your background" or "based on your experience" that makes personalization explicit."""
    else:
        personalization = ""
    
    personalization = f"<personalization>\n{personalization}\n</personalization>"
    sys_prompt = sys_prompt.replace("<personalization>", personalization)

    return sys_prompt
    
def build_input_from_history(message, history, certainty=True, personality_dict={}):

    parts = []
    parts.append({"role": "system", "content": get_db_sys_prompt(certainty=certainty, personality_dict=personality_dict)})
    
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
