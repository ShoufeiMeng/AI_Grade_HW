import streamlit as st

from langchain.memory import ConversationBufferMemory
from utils import qa_agent


st.title("AI Study Assistant: Upload PDFs and Get Smart Answers")

with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API key:：", type="password")
    st.markdown("[Get your OpenAI API key](https://platform.openai.com/account/api-keys)")

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

# uploaded_file = st.file_uploader("Upload your PDF：", type="pdf")
uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
question = st.text_input("Ask a question about the content of the PDFs")

# question = st.text_input("Ask a question about the content of your PDF", disabled=not uploaded_file)

if not openai_api_key:
    st.warning("🔑 Please enter your OpenAI API key.")
elif not uploaded_files:
    st.info("📄 Please upload at least one PDF file.")
elif not question:
    st.info("💬 Please enter a question about the content.")
else:
    try:
        with st.spinner("🤖 AI is thinking, please wait..."):
            response = qa_agent(openai_api_key, st.session_state["memory"],
                                uploaded_files, question)
        st.write("### ✅ Answer")
        st.write(response["answer"])
        st.session_state["chat_history"] = response["chat_history"]
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")

if "chat_history" in st.session_state:
    with st.expander("chat_history"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
