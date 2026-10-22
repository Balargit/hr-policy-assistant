
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from google import genai

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📚"
)

st.title("📚 HR Policy Assistant")
st.write("Ask questions about the available HR policies.")

# Read the API key from Streamlit Secrets
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# Fictional policies for the learning project
policies = [
    {
        "name": "Travel Policy",
        "text": "Hotel reimbursement is limited to 8000 rupees per night."
    },
    {
        "name": "Medical Policy",
        "text": "Employees must submit original invoices for medical reimbursement."
    },
    {
        "name": "Remote Work Policy",
        "text": "Employees may work remotely up to two days per week."
    },
    {
        "name": "Annual Leave Policy",
        "text": "Annual leave entitlement depends on the employee category."
    },
    {
        "name": "Leave Policy",
        "text": "Eligible employees receive 15 working days of paternity leave."
    }
]

@st.cache_resource
def load_model_and_embeddings():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    texts = [p["text"] for p in policies]
    embeddings = model.encode(
        texts,
        convert_to_tensor=True
    )
    return model, embeddings

with st.spinner("Loading policy documents..."):
    model, document_embeddings = load_model_and_embeddings()


def answer_question(question):
    query_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        query_embedding,
        document_embeddings
    )[0]

    best_index = int(scores.argmax().item())
    best_score = float(scores[best_index])
    policy = policies[best_index]

    # Provisional relevance threshold
    if best_score < 0.40:
        return (
            "I couldn't find sufficient supporting information.",
            None
        )

    prompt = f"""
You are an HR policy assistant.

Answer using only the supplied policy text.
Never invent facts or policy rules.
If the text does not support the answer, say so.

Source: {policy['name']}
Policy text: {policy['text']}
Question: {question}

Give a concise answer and identify the source policy.
"""

    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    return response.output_text, policy["name"]


# Display the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask an HR policy question...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching policies..."):
                answer, source = answer_question(question)

            st.markdown(answer)

            if source:
                st.caption(f"Source: {source}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception:
            st.error(
                "The answer could not be generated. "
                "Please check the app configuration and logs."
            )
