import streamlit as st

st.title("Dobble Bot")
st.write("Upload images → generate a Dobble-style deck (coming next).")

files = st.file_uploader(
    "Upload images (PNG/JPG). Tip: 10–30 is great to start.",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if files:
    st.success(f"Loaded {len(files)} images.")
