import os
import gradio as gr
from openai import AsyncOpenAI
import asyncio

from logger import log_event
from chat_helpers import build_input_from_history

oclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def respond(message, history):
    text_input = build_input_from_history(message, history)
    kwargs = dict(
        model="gpt-4.1",
        input=text_input,
        temperature=0,
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

async def chat_driver(user_message, messages_history, _pid):
    messages_history = messages_history or []
    base = messages_history + [{"role": "user", "content": user_message}]
    assistant_text = ""

    asyncio.create_task(log_event(_pid, "chat_user",
                                 {"text": user_message}))

    async for chunk in respond(user_message, messages_history):
        assistant_text = chunk
        yield base + [{"role": "assistant", "content": assistant_text}], ""

    asyncio.create_task(log_event(_pid, "chat_assistant",
                                 {"text": assistant_text}))

def get_params_from_request(request: gr.Request):
    try:
        qp = request.query_params or {}
        def _get(key, default=""):
            return qp.get(key, default) if hasattr(qp, "get") else (qp[key] if key in qp else default)

        pid = _get("pid") or _get("response_id") or _get("ResponseID") or _get("id") or "anon"

        return pid
    except Exception:
        return "anon"

with gr.Blocks(title="StormShield Risk Management Bot", theme="soft") as demo:

    pid_state = gr.State("anon")

    with gr.Column(visible=True) as app_view:
        gr.Markdown("# StormShield Risk Management Bot")
        gr.Textbox(
            label="Try one of these prompts:",
            value=(
                "- How can sensors misread water levels in the first months?\n"
                "- Would StormShield barriers fit smoothly with the older seawall?\n"
                "- Are city computers too old to run StormShield?\n"
                "- Evaluate the expert claims and the recommended budget."
            ),
            lines=4,
            max_lines=4,
            interactive=False,
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
            fn=get_params_from_request,
            inputs=[],
            outputs=[pid_state],
        )

        ev = send_btn.click(
            chat_driver,
            inputs=[chat_input, chatbot, pid_state],
            outputs=[chatbot, chat_input]
        )

        ev2 = chat_input.submit(
            chat_driver,
            inputs=[chat_input, chatbot, pid_state],
            outputs=[chatbot, chat_input]
        )


if __name__ == "__main__":
    demo.launch(share=True)
