import os
import gradio as gr
from openai import AsyncOpenAI
import asyncio

from logger import log_event, load_chat_history
from chat_helpers import build_input_from_history

SUGGESTED_PROMPTS = [
    "How can sensors misread water levels in the first months?",
    "Would StormShield barriers fit smoothly with the older seawall?",
    "Are city computers too old to run StormShield?",
    "Evaluate the expert claims and the recommended budget.",
]

CUSTOM_CSS = """
.suggested-prompts .gr-row {
    gap: 0.5rem;
}
.suggested-prompts .gr-button {
    width: 100%;
    font-size: 0.9rem;
    padding: 0.45rem 0.6rem;
}
"""

oclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def respond(message, history, competence, personalization):
    text_input = build_input_from_history(message, history, competence=competence, personalization=personalization)
    kwargs = dict(
        model="gpt-4.1",
        input=text_input,
        temperature=0,
        max_output_tokens=512,
        top_p=0.0,
        text={
            "verbosity": "medium"
        }
    )

    buffer = []
    async with oclient.responses.stream(**kwargs) as stream:
        async for event in stream:
            if event.type == "response.output_text.delta":
                buffer.append(event.delta)
                yield "".join(buffer)

        final = await stream.get_final_response()
        final_text = getattr(final, "output_text", None)
        if final_text and (not buffer or final_text != "".join(buffer)):
            yield final_text

async def chat_driver(user_message, messages_history, _pid, competence, personalization):
    messages_history = messages_history or []
    base = messages_history + [{"role": "user", "content": user_message}]
    assistant_text = ""

    asyncio.create_task(log_event(_pid, "chat_user",
                                 {"text": user_message}))

    async for chunk in respond(user_message, messages_history, competence=competence, personalization=personalization):
        assistant_text = chunk
        yield base + [{"role": "assistant", "content": assistant_text}], ""

    asyncio.create_task(log_event(_pid, "chat_assistant",
                                 {"text": assistant_text}))

async def init_from_request(request: gr.Request):
    pid, competence, personalization = get_params_from_request(request)
    competence_flag = (competence == "1")
    personalization_flag = (personalization == "1")
    history = await asyncio.to_thread(load_chat_history, pid)

    asyncio.create_task(log_event(pid, "session_start", {"competence": competence_flag, "personalization": personalization_flag}))

    return pid, competence_flag, personalization_flag, history

def get_params_from_request(request: gr.Request):
    try:
        qp = request.query_params or {}
        def _get(key, default=""):
            return qp.get(key, default) if hasattr(qp, "get") else (qp[key] if key in qp else default)

        pid = _get("pid") or _get("response_id") or _get("ResponseID") or _get("id") or "anon"
        personalization = _get("pers", "1")
        competence = _get("comp", "1")

        return pid, competence, personalization
    except Exception:
        return "anon", True, True

with gr.Blocks(title="StormShield Risk Management Bot", theme="soft", css=CUSTOM_CSS) as demo:

    pid_state = gr.State("anon")
    competence_state = gr.State(True)
    personalization_state = gr.State(True)
    prompt_states = []
    prompt_buttons = []

    with gr.Column(visible=True) as app_view:
        gr.Markdown("# StormShield Risk Management Bot")
        with gr.Column(elem_classes=["suggested-prompts"]):
            gr.Markdown("**Try one of these prompts:**")
            for idx in range(0, len(SUGGESTED_PROMPTS), 2):
                with gr.Row():
                    for prompt in SUGGESTED_PROMPTS[idx:idx + 2]:
                        prompt_state = gr.State(prompt)
                        prompt_states.append(prompt_state)
                        prompt_buttons.append(
                            gr.Button(prompt, variant="secondary", min_width=0)
                        )
        chatbot = gr.Chatbot(type="messages", resizable=True, label=None, height=600, show_label=False)

        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Ask me about StormShield!",
                scale=8,
                autofocus=False,
                container=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        demo.load(
            fn=init_from_request,
            inputs=[],
            outputs=[pid_state, competence_state, personalization_state, chatbot],
        )

        ev = send_btn.click(
            chat_driver,
            inputs=[chat_input, chatbot, pid_state, competence_state, personalization_state],
            outputs=[chatbot, chat_input]
        )

        ev2 = chat_input.submit(
            chat_driver,
            inputs=[chat_input, chatbot, pid_state, competence_state, personalization_state],
            outputs=[chatbot, chat_input]
        )

        for btn, prompt_state in zip(prompt_buttons, prompt_states):
            btn.click(
                chat_driver,
                inputs=[prompt_state, chatbot, pid_state, competence_state, personalization_state],
                outputs=[chatbot, chat_input],
            )


if __name__ == "__main__":
    demo.launch(share=True)
