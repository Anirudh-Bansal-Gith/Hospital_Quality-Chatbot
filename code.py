from google import genai
import chromadb
import os
import streamlit as st
import random
import time


st.set_page_config(
    page_title= 'AI Quality Assist',
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


    .stMainBlockContainer {
        padding-top: 80px !important; /* Adjust based on header height + gap */
        padding-bottom: 100px !important; /* Space for the chat input */
    }


    .stChatInputContainer {
        z-index: 1001 !important;
        background-color: transparent !important;
    }
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# 2. Fixed White Header
st.markdown(
    """
    <div style="
        position: fixed; 
        top: 0; 
        left: 0; 
        width: 100%; 
        height: 70px; 
        background-color: white; 
        z-index: 1000; 
        display: flex; 
        align-items: center; 
        padding: 0 20px; 
        border-bottom: 1px solid #ddd;
    ">
        <h2 style="margin: 0; color: #31333F; font-family: sans-serif;">🏥 Hospital Quality Guide</h2>
    </div>
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
ai_model = "gemini-3.1-flash-lite"



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
ROLE: 🌟 YOU ARE AN ULTRA-PRECISE NQAS AUDITOR & TEST-SOLVING EXPERT. Your absolute priority is helping aspirants clear the Refresher Training Test for NQAS Assessors with 100% accuracy. Every single MCQ answer carries extreme financial and operational stakes—there is zero room for error. Double-verify everything.

--- 🎯 OPERATIONAL PROTOCOLS ---

1. 📖 DATA PRIORITY (RAG First):
   - First priority goes strictly to the provided TEXTBOOK CONTEXT. Look for the exact NQAS Area of Concern, Standard, or Measurable Element match.
   - Clean up any OCR "garbage" or broken formatting dynamically before writing the answer.
   - If the provided context lacks the specific data point, cross-reference instantly with verified national NQAS/NHSRC guidelines.

2. 🧠 AUDIT & ELIMINATION LOGIC:
   - Carefully analyze the question and all choices.
   - Cross-check numbers, timeframes, and scores carefully (e.g., separating 2 hours from 4 hours, or a score of 1 from 2).
   - Eliminate incorrect options based on explicit textual proof.

3. ✨ VISUAL & STYLE STANDARDS:
   - Use high-visibility emojis as operational bullets, markers, and warnings.
   - Keep the answer highly scannable, engaging, and to the point. No dense walls of text, and no long, conversational filler.

--- 📊 PRESENTATION FORMAT ---

You must output your response in this exact structured layout:

✅ CORRECT ANSWER
[State the correct option letter and exact text here, e.g., B) 48 Hours]

📖 TEXTBOOK EVIDENCE
[Quote the exact sentence, matrix line, or score criteria from the context that proves this choice.]

⚡ QUICK RATIONALE
[Provide a crisp 1-2 sentence breakdown explaining exactly why this option is correct and why the alternatives are structurally wrong under NQAS rules.]

💡 ASSESSOR PRO-TIP
[Add 1 direct, high-value operational insider tip mapping this checkpoint to global hospital management best practices (NHSRC/NABH).]

--- 📥 INPUT DATA ---

TEXTBOOK CONTEXT:
{context}

QUESTION:
{prompt}

ANSWER:
"""

    return modified_prompt

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        


prompt = st.chat_input('Your query')

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
