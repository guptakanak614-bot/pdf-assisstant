import streamlit as st
import PyPDF2
import re
import random
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")
st.title("📘 chat with pdf")

# ---------------- SESSION STATE ----------------
if "notes" not in st.session_state:
    st.session_state.notes = []

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- HELPERS ----------------
def make_chunks(text, lines_per_chunk=3):
    lines = text.split("\n")
    chunks, temp = [], []

    for line in lines:
        line = line.strip()
        if line:
            temp.append(line)

        if len(temp) == lines_per_chunk:
            chunks.append(" ".join(temp))
            temp = []

    if temp:
        chunks.append(" ".join(temp))

    return chunks


def highlight(text, question):
    for word in question.split():
        if len(word) > 3:
            text = re.sub(
                fr"({word})",
                r"**\1**",
                text,
                flags=re.IGNORECASE
            )
    return text


def generate_mcq(text):
    words = [w for w in text.split() if len(w) > 6]
    if len(words) < 4:
        return None

    answer = random.choice(words)
    options = random.sample(words, min(4, len(words)))
    if answer not in options:
        options[0] = answer

    random.shuffle(options)
    return answer, options


# ---------------- PDF UPLOAD ----------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

pages, chunks, chunk_page = [], [], []

if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)

    for pno, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        pages.append(text)

        for ch in make_chunks(text):
            if len(ch) > 120:
                chunks.append(ch)
                chunk_page.append(pno)

    # ---------------- LAYOUT ----------------
    col1, col2, col3 = st.columns([1, 1.2, 0.8])

    # -------- LEFT : PDF --------
    with col1:
        st.subheader("📄 PDF (Original)")
        p = st.number_input("Select Page", 1, len(pages), 1)
        st.text(pages[p - 1])

    # -------- CENTER : STUDY MODE --------
    with col2:
        st.subheader("📚 Study Mode")

        mode = st.radio("Mode", ["Answer", "Summary", "MCQs"])
        question = st.text_input("Ask question / topic")

        if question:
            st.session_state.history.append(question)

            vectorizer = TfidfVectorizer(stop_words="english")
            vectors = vectorizer.fit_transform(chunks + [question])

            sim = cosine_similarity(vectors[-1], vectors[:-1])[0]
            top = sim.argsort()[-3:][::-1]

            final_text = ""

            if mode == "Answer":
                st.success("Answer:")
                for i in top:
                    text = highlight(chunks[i], question)
                    st.markdown(text)
                    st.caption(f"📄 Page {chunk_page[i] + 1}")
                    final_text += f"{chunks[i]} (Page {chunk_page[i]+1})\n\n"
                    st.divider()

            elif mode == "Summary":
                st.success("Exam Ready Summary:")
                summary = " ".join([chunks[i] for i in top])
                st.markdown(summary[:900] + "...")
                final_text = summary

            else:
                st.success("MCQs for Practice:")
                for i in top:
                    mcq = generate_mcq(chunks[i])
                    if mcq:
                        ans, opts = mcq
                        st.markdown("**Question:** Choose correct term")
                        for o in opts:
                            st.write("◻️", o)
                        st.caption(f"✔ Answer: {ans}")
                        st.divider()

            if mode != "MCQs":
                if st.button("💾 Save to Notes"):
                    st.session_state.notes.append(
                        f"[{datetime.now().strftime('%d-%m-%Y %H:%M')}]\nQ: {question}\n{final_text}"
                    )
                    st.success("Saved to Notes")

    # -------- RIGHT : NOTES & HISTORY --------
    with col3:
        st.subheader("📝 Notes")

        if st.session_state.notes:
            notes_text = "\n\n".join(st.session_state.notes)
            st.download_button(
                "⬇ Download Notes",
                notes_text,
                file_name="pdf_notes.txt"
            )
        else:
            st.info("No notes yet")

        st.subheader("🕘 Question History")
        for q in st.session_state.history[-5:][::-1]:
            st.write("•", q)