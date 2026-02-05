import os
import sys

from src.rag_chatbot.ui.app_utils import load_params_from_txt
from src.rag_chatbot.ui.theme import get_css, get_theme
from datetime import datetime
from dotenv import load_dotenv
from gradio import Blocks, Row, Column, Image, Button, Markdown, Chatbot, Textbox, State, LikeData
from gradio.components import ChatMessage
from uuid import uuid4
from typing import List, Tuple
from src.rag_chatbot.rag.RAG_bot import generate_contextualized_response


# Optionally, load .env from current directory to override/add UI-specific vars
load_dotenv('.env')


# Load externally set content (allows for easy rebranding)
content = load_params_from_txt("ui/res/content.txt")

def generate_placeholder_reply(message: str) -> str:
    """Simple placeholder reply. Replace this with an LLM call."""
    query_dict = {"question": message}
    response_dict = generate_contextualized_response(query_dict)
    response = response_dict["answer"]
    return response


def new_chat(state):
    state["history"] = []
    state["chat_uuid"] = str(uuid4())
    return state, state.get("history")

def browse_history():
    pass


def respond(state, txt):
    state.get("history").append(
        ChatMessage(role="user",
                    content=txt,
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "chat_id": state.get("chat_uuid", "unknown"),
                        "message_id": str(uuid4())
                        }
                    )
    )
    state.get("history").append(
        ChatMessage(role="assistant",
                    content=generate_placeholder_reply(txt)
                    )
    )
    return state, "", state.get("history")

def like(evt: LikeData):
    print("User liked the response")
    print(evt.index, evt.liked, evt.value)

def build_ui() -> Blocks:
    with Blocks(title=content.get('CHATBOT_NAME', 'Chat Assistant')) as block:
        with Row(elem_id="top-row", max_height="200px"):
            Markdown(
                f"# {content.get('CHATBOT_NAME', 'Chat Assistant')} for {content.get('COMPANY_NAME', 'Rapid Deployment')}\n"
                "Welcome to your AI-powered chat assistant. Start a new chat or continue an existing conversation."
            )
           

        # Control buttons (horizontal)
        with Row(elem_id="controls-row"):
            with Column(scale=1, min_width=0):
                with Row(elem_classes="controls"):
                    btn_new = Button("New Chat", elem_id="btn-new", variant="primary")
                    btn_browse = Button("Browse Chat History", elem_id="btn-browse", variant="primary")

        # Chatbot area
        with Row(elem_id="chat-row"):
            with Column(scale=1):
                chatbot = Chatbot(label=content.get("CHATBOT_NAME", "Assistant"), elem_id="chatbot")
                chatbot.like(like)
                with Row(elem_id="input-row"):
                    txt = Textbox(show_label=False, placeholder="Type your message and press Enter...")

        # Wire interactions
        state = State({
            "chat_uuid": str(uuid4()),
            "history": []
        })  # holds chat history as list of ChatMessage

        btn_new.click(
            fn=new_chat,
            inputs=[state],
            outputs=[state, chatbot],
        )

        # Browse chat history
        btn_browse.click(
            fn=browse_history,
            inputs=[],
            outputs=[],
        )

        # send via button or Enter in textbox
        txt.submit(respond, [state, txt], [state, txt, chatbot])

    return block


def main():
    app = build_ui()
    app.launch(
        # auth=("username", "password"),
        css=get_css(),
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        theme=get_theme(),
        )


if __name__ == "__main__":
    main()
