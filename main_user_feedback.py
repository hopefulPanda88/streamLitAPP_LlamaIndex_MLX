import os
import tempfile

import yaml
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st

from trubrics.integrations.streamlit import FeedbackCollector

from utils import LocalLLMOnMLX
from utils.chatAPP.fused_chat import generate_engine_from_documents
from utils.chatAPP.simpleChat import simple_chat

st.set_page_config(
    page_title="With Feedback Demo",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📝 Chat with Feedback")
"""
This example shows two mode: with and without RAG technique. In addition to those, it embeds one 
method to collect feedback from users with [Trubrics]( https://trubrics.streamlit.app/).
"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you? Leave feedback to help me improve!"}
    ]
if "response" not in st.session_state:
    st.session_state["response"] = None

if "current_model" not in st.session_state:
    st.session_state["current_model"] = ""

if "model" not in st.session_state:
    st.session_state["model"] = None

if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = None

if "logged_prompt" not in st.session_state:
    st.session_state.logged_prompt = None

if "feedback_key" not in st.session_state:
    st.session_state.feedback_key = 0

if "query_engine" not in st.session_state:
    st.session_state["query_engine"] = None

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "normal"

if "config_data" not in st.session_state:
    with open("./config/model.yaml", 'r') as f:
        data = yaml.safe_load(f)
    st.session_state["config_data"] = data

# sidebar
with st.sidebar:
    st.title("Model Selection")
    model_list = []
    for i in st.session_state["config_data"]["models"]:
        model_list.append(i)
    option = st.selectbox(
        'Model List',
        model_list)

    st.session_state["if_selected_model"] = st.button("Load the LLM", type="primary")
    if st.session_state["if_selected_model"]:
        if option == st.session_state["current_model"]:
            st.write(f"Model {option} is already loaded and running! :dog:")
        else:
            local_model_path = st.session_state["config_data"]["models"][option]["path"]
            with st.spinner(f"Loading model {option} from {local_model_path}, please wait..."):
                llm = LocalLLMOnMLX(local_model_path, local_model_path)
            st.session_state["current_model"] = option
            st.session_state["model"] = llm
            Settings.llm = llm
            st.write(f"Current model is {option}!")
            st.toast(f"Model {option} has been loaded!", icon="🤖")
    if st.session_state["current_model"]:
        st.write(" ")
        st.markdown("***")
        st.write("*Your could either go right  to chat with LLM directly, "
                 "or chat with your files after uploading them from below.*")
        st.markdown("***")

        uploaded_files = st.file_uploader("Choose a file or multiple PDF files", type='pdf', accept_multiple_files=True)
        pdf_files = []
        if uploaded_files:
            for u_file in uploaded_files:
                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, u_file.name)
                with open(path, "wb") as f:
                    f.write(u_file.getvalue())
                pdf_files.append(path)

        if len(pdf_files) > 0:
            p_clicked = st.button("Process files", type="primary")
            if p_clicked:
                if st.session_state["embedding_model"] is None:
                    with st.spinner("Load the embedding model for the first time..."):
                        st.session_state["embedding_model"] = HuggingFaceEmbedding(
                            st.session_state["config_data"]["embedding_models"]["UAELarge"]["path"], max_length=512)
                        Settings.embed_model = st.session_state["embedding_model"]
                        st.success("embedding model loaded!")
                with st.spinner("Now generate the engine for you to query or chat..."):
                    st.session_state["query_engine"] = generate_engine_from_documents(
                        pdf_files, llm=st.session_state["model"], embedding_model=st.session_state["embedding_model"])
                    st.success("Engine has been set!")
                st.session_state.chat_mode = "files"
                st.toast("We are converting to the chat with file mode!")

messages = st.session_state.messages
for msg in messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Tell me a joke about sharks"):
    if st.session_state["current_model"] == "":
        st.error("No model has been loaded yet! :red_circle:")
    else:
        messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if st.session_state.chat_mode == "files":
            question = messages[len(messages) - 1]["content"]
            response = (st.session_state["query_engine"].query(question)).response
        else:
            response = simple_chat(messages=messages, llm=st.session_state["model"])
        st.session_state["response"] = response
        with st.chat_message("assistant"):
            messages.append({"role": "assistant", "content": st.session_state["response"]})
            st.write(st.session_state["response"])

    if st.session_state["response"]:
        # feedback_option = "faces" if st.toggle(label="`Thumbs` ⇄ `Faces`", value=False) else "thumbs"

        collector = FeedbackCollector(
            project="default",
            email=st.secrets.TRUBRICS.TRUBRICS_EMAIL,
            password=st.secrets.TRUBRICS.TRUBRICS_PASSWORD,
        )
        st.session_state.logged_prompt = collector.log_prompt(
            config_model={"model": st.session_state["current_model"]},
            prompt=prompt,
            generation=st.session_state["response"]
        )
        if st.session_state.logged_prompt:
            user_feedback = collector.st_feedback(
                component="default",
                feedback_type="faces",
                open_feedback_label="[Optional] Provide additional feedback",
                model=str(st.session_state.logged_prompt.config_model),
                prompt_id=st.session_state.logged_prompt.id,
                key=str(st.session_state.feedback_key),
                align="flex-start"
            )
            st.write(user_feedback)

