import os
import gradio as gr
from openai import AsyncOpenAI
import asyncio

from logger import log_event, load_chat_history
from chat_helpers import build_input_from_history

QUALTRICS_BRIDGE_JS = r"""
function(GradioApp) {
  console.log("✅ qualtrics bridge JS loaded");

  let busy = false;
  let observer = null;
  let idleTimer = null;

  function notifyBusy() {
    if (busy) return;
    busy = true;
    console.log("[Gradio] -> parent: chat_busy");
    if (window.parent) {
      window.parent.postMessage({ type: "chat_busy" }, "*");
    }
  }

  function notifyIdle() {
    if (!busy) return;
    busy = false;
    console.log("[Gradio] -> parent: chat_idle");
    if (window.parent) {
      window.parent.postMessage({ type: "chat_idle" }, "*");
    }
  }

  function ensureObserver() {
    if (observer) return;

    // Try to locate the chatbot container
    const chat =
      document.getElementById("stormshield-chatbot") ||
      document.querySelector("#stormshield-chatbot") ||
      document.querySelector('[data-testid="chatbot"]');

    console.log("[Gradio] Chat container for observer:", chat);
    if (!chat) {
      console.log("[Gradio] No chat container found for MutationObserver");
      return;
    }

    observer = new MutationObserver(function (mutations) {
      if (!busy) return;

      let hasNewContent = false;
      for (const m of mutations) {
        if (m.addedNodes && m.addedNodes.length > 0) {
          hasNewContent = true;
          break;
        }
      }
      if (!hasNewContent) return;

      // Each time the assistant output changes, reset a timer.
      // When there are no more changes for 1500ms, we assume the answer is done.
      if (idleTimer) {
        clearTimeout(idleTimer);
      }
      idleTimer = setTimeout(function () {
        console.log("[Gradio] No chat updates for 1500ms, marking idle");
        notifyIdle();
      }, 1500);
    });

    observer.observe(chat, { childList: true, subtree: true });
    console.log("[Gradio] MutationObserver attached");
  }

  window.addEventListener("message", function (event) {
    console.log("[Gradio] postMessage received:", event.data, "from", event.origin);

    if (!event.data || event.data.type !== "qualtrics_prompt") {
      return;
    }

    const text = event.data.text || "";
    console.log("[Gradio] qualtrics_prompt text:", text);
    if (!text) return;

    // Prepare observer and mark busy before sending
    ensureObserver();
    notifyBusy();

    // Find textbox
    const textbox =
      document.querySelector('textarea[placeholder="Ask me about StormShield!"]') ||
      document.querySelector("textarea") ||
      document.querySelector('[contenteditable="true"]');

    // Find Send button
    const sendBtn = Array.prototype.find.call(
      document.querySelectorAll("button"),
      function (btn) {
        return btn.innerText.trim() === "Send";
      }
    );

    console.log("[Gradio] textbox:", textbox, "sendBtn:", sendBtn);

    if (!textbox || !sendBtn) {
      console.log("❌ Could not find textbox or send button");
      // if we can't actually send, immediately go back to idle so parent doesn't stay frozen
      notifyIdle();
      return;
    }

    // Fill textbox
    if (textbox.tagName.toLowerCase() === "textarea" || textbox.tagName.toLowerCase() === "input") {
      textbox.value = text;
      textbox.dispatchEvent(new Event("input", { bubbles: true }));
    } else if (textbox.getAttribute("contenteditable") === "true") {
      textbox.innerText = text;
      textbox.dispatchEvent(new Event("input", { bubbles: true }));
    }

    console.log("✏️ Filled textbox with:", text);

    // Click Send – MutationObserver will watch for answer completion
    sendBtn.click();
    console.log("✅ Clicked Send button");
  });
}
"""


oclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def respond(message, history, competence, personality_dict):
    text_input = build_input_from_history(message, history, competence=competence, personality_dict=personality_dict)
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

async def chat_driver(user_message, messages_history, _pid, competence, personality_dict):
    if not user_message:
        yield messages_history, ""
        return

    messages_history = messages_history or []
    base = messages_history + [{"role": "user", "content": user_message}]
    assistant_text = ""

    asyncio.create_task(log_event(_pid, "chat_user",
                                 {"text": user_message}))

    async for chunk in respond(user_message, messages_history, competence=competence, personality_dict=personality_dict):
        assistant_text = chunk
        yield base + [{"role": "assistant", "content": assistant_text}], ""

    asyncio.create_task(log_event(_pid, "chat_assistant",
                                 {"text": assistant_text}))
    
    yield base + [{"role": "assistant", "content": assistant_text}], ""

async def init_from_request(request: gr.Request):
    # 1. Parse params including the new 'user_prompt'
    pid, competence, personality_dict = get_params_from_request(request)
    
    competence_flag = (competence == "1")
    personalization = bool(personality_dict)
    
    # 2. Load History (Crucial: This ensures history persists even when iframe reloads)
    history = await asyncio.to_thread(load_chat_history, pid)

    asyncio.create_task(log_event(pid, "session_start", {"competence": competence_flag, "personalization": personalization}))

    # 3. Return state AND the user_prompt found in URL to the hidden input
    return pid, competence_flag, personality_dict, history

def get_params_from_request(request: gr.Request):
    try:
        qp = request.query_params or {}
        def _get(key, default=""):
            return qp.get(key, default) if hasattr(qp, "get") else (qp[key] if key in qp else default)

        pid = _get("pid") or _get("response_id") or _get("ResponseID") or _get("id") or "anon"
        competence = _get("comp", "1")
        nickname = _get("nickname", "")
        age = _get("age", "")
        education = _get("education", "")
        work = _get("work", "")
        hobbies = _get("hobbies", "")

        if nickname and age and education and work and hobbies:
            personality_dict = {
                "nickname": nickname,
                "age": age,
                "education": education,
                "work": work,
                "hobbies": hobbies
            }
        else:
            personality_dict = {}

        return pid, competence, personality_dict
    except Exception:
        return "anon", True, {}, ""

with gr.Blocks(title="StormShield Risk Management Bot", theme="soft", js=QUALTRICS_BRIDGE_JS) as demo:

    pid_state = gr.State("anon")
    competence_state = gr.State(True)
    personality_dict = gr.State({})
    prompt_states = []
    prompt_buttons = []

    with gr.Column(visible=True) as app_view:
        gr.Markdown("# StormShield Risk Management Bot")
        chatbot = gr.Chatbot(type="messages", resizable=True, label=None, height=600, show_label=False, elem_id="stormshield-chatbot")

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
            outputs=[pid_state, competence_state, personality_dict, chatbot]
        )

        send_btn.click(
            chat_driver,
            inputs=[chat_input, chatbot, pid_state, competence_state, personality_dict],
            outputs=[chatbot, chat_input]
        )

        chat_input.submit(
            chat_driver,
            inputs=[chat_input, chatbot, pid_state, competence_state, personality_dict],
            outputs=[chatbot, chat_input]
        )

if __name__ == "__main__":
    demo.launch(share=True)