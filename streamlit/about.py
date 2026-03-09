import streamlit as st
import streamlit_shortcuts as shortcuts

with open("READme.md", "r") as f:
    content = "".join(f.readlines())
    PARAGRAPHS = content.split("\n\n---\n\n")
    PARAGRAPHS[0] = PARAGRAPHS[0][PARAGRAPHS[0].find("##"):]

if "idx" not in st.session_state:
    st.session_state.idx = 0


def go_next():
    st.session_state.idx = min(st.session_state.idx + 1, len(PARAGRAPHS) - 1)


def go_prev():
    st.session_state.idx = max(st.session_state.idx - 1, 0)


lines = PARAGRAPHS[st.session_state.idx].split("\n\n")

header = lines[0]

body = "\n\n".join(lines[1:])

st.markdown(header)

st.markdown(body)
st.write("")

c0, c1, c2, _ = st.columns([0.85, 0.045, 0.045, 0.05], gap="xxsmall")

with c0:
    st.markdown(f"*:gray[**{st.session_state.idx + 1} / {len(PARAGRAPHS)}**"
                f"⠀-⠀press arrow keys or enter to continue⠀**→**]*")

with c1:
    st.button("∧", on_click=go_prev, disabled=(st.session_state.idx == 0), key="prev_btn")

with c2:
    st.button("∨", on_click=go_next, disabled=(st.session_state.idx == len(PARAGRAPHS) - 1), key="next_btn")

st.markdown("""
<style>
div.stButton > button {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 0.25rem 0.4rem;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

shortcuts.add_shortcuts(prev_btn=["arrowup"], next_btn=["arrowdown", "enter"])