from google import genai
import chromadb
import os
import streamlit as st
import random
import time


st.set_page_config(
    page_title= 'AI Quality Assist',
    page_icon="logo.jpeg", 
    layout="wide"
)

hide_st_style = """
    <style>
    html, body, #root, [data-testid="stAppViewContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stHeader"], 
    [data-testid="stDecoration"], 
    footer {
        display: none !important;
    }

    .block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)


st.markdown(
    """
    <div style="
        position: fixed; 
        top: 30; 
        left: 0; 
        width: 100%; 
        height: 70px; 
        background-color: red; 
        z-index: 1000; 
        display: flex; 
        align-items: center; 
        padding: 0 20px; 
        border-bottom: 1px solid #ddd;
    ">
        <h2 style="margin: 0; color: #31333F; font-family: sans-serif;">🏥 Hospital Quality Guide</h2>
    </div>
    
    <style>
    .stApp {
        margin-top: 0px;
    }
    .stChatInputContainer {
        z-index: 1001 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

keys = [
    st.secrets['GEMINI_KEY_48B'], st.secrets['GEMINI_KEY_48C'], st.secrets['GEMINI_KEY_865'], st.secrets['GEMINI_KEY_866'],
         st.secrets['GEMINI_KEY_867'], st.secrets['GEMINI_KEY_868'], st.secrets['GEMINI_KEY_869'],st.secrets['GEMINI_KEY_870'] 
        ]

@st.cache_resource
def get_genai_client(api_key):
    return genai.Client(api_key=api_key)


@st.cache_resource
def get_db_collection():
    db_client = chromadb.PersistentClient(path="textbook_db")
    return db_client.get_or_create_collection(name="textbook_collection")


if 'assigned_key' not in st.session_state:
    st.session_state.assigned_key = random.choice(keys)

client = get_genai_client(st.session_state.assigned_key)
collection = get_db_collection()
ai_model = "gemini-2.5-flash-lite"



if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(model=ai_model)


def modify_prompt(prompt):

    r = client.models.embed_content(
        model="gemini-embedding-2-preview", 
        contents = prompt, config={'output_dimensionality': 768}
    )
    query_vector = r.embeddings[0].values
    results = collection.query(
        query_embeddings=[query_vector], 
        n_results=5)

    context_list = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        source_info = f"[BOOK: {meta.get('book', 'Unknown')} | PAGE: {meta.get('page_num', 'N/A')}]"
        context_list.append(f"{source_info}\n{doc}")

    context = "\n\n---\n\n".join(context_list)

    modified_prompt = f"""
Role: Expert Hospital Quality Guide and Medical Standards Editor. 

--- FORMATTING RULES (STRICT) ---
1. NO MARKDOWN: You are strictly forbidden from using asterisks (**), underscores (_), or hashes (#).
2. PLAIN TEXT ONLY: The entire response must be standard plain text. No bolding, even for headers.
3. HEADERS: Use simple CAPITAL LETTERS for headers (e.g., AI SUMMARY).

--- DECISION LOGIC ---
1. IF THE ANSWER IS FROM THE TEXTBOOK:
   - Provide the OFFICIAL DOCUMENT SNIPPET (Rewritten/Cleaned).
   - Provide the AI SUMMARY AND EXPLANATION.
2. IF THE ANSWER IS A SUMMARY OF PREVIOUS CHAT OR GENERAL KNOWLEDGE:
   - Provide ONLY the AI SUMMARY AND EXPLANATION.
   - Do NOT include a snippet section.

--- CONTENT INSTRUCTIONS ---
- SOURCE CLEANING: If using a snippet, rewrite it. Delete all OCR garbage like 'OB', 'SI', 'me f2.1'. Fix broken spacing.
- UNIFORMITY: Ensure every line has the same font weight and visual style. No half-bold errors.
- LANGUAGE: Match the user's language (English/Hindi/Hinglish).

TEXTBOOK CONTEXT :
{context}

QUESTION:
{prompt}

ANSWER:
"""

    return modified_prompt

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        


prompt = st.chat_input()

if prompt:
    with st.chat_message('user'):
        st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner('Generating Response...'): 
        try:

            final_prompt = modify_prompt(prompt)

            response = st.session_state.chat_session.send_message(final_prompt)
            

            with st.chat_message('ai'):
                st.write(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

            row_response = [prompt,response.text, st.session_state.assigned_key]

                    
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if "PerDay" in str(e):
                    old_key = st.session_state.user_assigned_key
                    new_key = random.choice(keys)
                    while new_key == old_key:
                        new_key= random.choice(keys)
                    st.session_state.assigned_key = new_key
                    client = genai.Client(api_key=new_key)
                    st.session_state.chat_session = client.chats.create(model=ai_model)

                    st.rerun()
                else:
                    print("   ⏳ Minute limit hit (429). Sleeping for 60 seconds...")
                    time.sleep(60)
            else:
                st.error(f"Error: {e}")
