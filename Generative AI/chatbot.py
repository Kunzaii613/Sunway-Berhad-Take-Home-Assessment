import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import HuggingFaceHub

load_dotenv()

# Clear history after refresh
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Virtual Assistant", page_icon="😭")
st.title("Virtual Assistant")


# Get Response
def get_response(query, chat_history):
    template = """
    Your are a helpful assistant chatbot, kindly answer the following question:

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    prompt_str = prompt.format(chat_history=chat_history, user_question=query)

    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct")
    llm.client.api_url = (
        "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    )

    result = llm.generate([prompt_str])

    # Debugging
    print(result)
    generated_text = (
        result.generations[0] if result.generations else "No response generated."
    )

    return generated_text


# Showing Chat Log
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# User Input
user_query = st.chat_input("Message Chatbot")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history)
        st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(ai_response))
