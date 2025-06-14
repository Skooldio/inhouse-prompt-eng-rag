import streamlit as st
import asyncio
import time
from dotenv import load_dotenv
from chat_agent.llm_retrieval import get_assistant_response

load_dotenv()

st.title("Chat with an LLM")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Let's start chatting! 👇"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


async def initialize_chat():
    """Initialize the chat by displaying the initial message."""
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = await get_assistant_response(prompt)
            # Simulate stream of response with milliseconds delay
            # for chunk in assistant_response.content.split():
            #     full_response += chunk + " "
            #     time.sleep(0.05)
            #     # Add a blinking cursor to simulate typing
            #     message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(assistant_response.content)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response.content}
        )


async def main():
    await initialize_chat()


if __name__ == "__main__":
    asyncio.run(main())
