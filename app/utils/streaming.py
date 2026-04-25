import streamlit as st


def stream_response(llm, prompt):
    placeholder = st.empty()
    full_response = ""

    for chunk in llm.stream(prompt):
        full_response += chunk
        placeholder.markdown(full_response + "▌")

    placeholder.markdown(full_response)
    return full_response