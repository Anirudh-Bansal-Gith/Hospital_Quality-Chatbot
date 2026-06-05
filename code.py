import streamlit as st
import random
import time
import re
from google import genai
import chromadb

# --- 1. UI SETUP & CSS ---
st.set_page_config(page_title='AI Quality Assist', layout="wide")

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
        padding-top: 80px !important; 
        padding-bottom: 100px !important; 
    }
    .stChatInputContainer {
        z-index: 1001 !important;
        background-color: transparent !important;
    }
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

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

# --- 2. INITIALIZATION & KEY ROTATION ---
keys = [
    st.secrets['GEMINI_KEY_48B'], st.secrets['GEMINI_KEY_48C'], 
    st.secrets['GEMINI_KEY_865'], st.secrets['GEMINI_KEY_866'],
    st.secrets['GEMINI_KEY_867'], st.secrets['GEMINI_KEY_868'], 
    st.secrets['GEMINI_KEY_869'], st.secrets['GEMINI_KEY_870'] 
]

if 'assigned_key' not in st.session_state:
    st.session_state.assigned_key = random.choice(keys)

@st.cache_resource
def get_db_collection():
    db_client = chromadb.PersistentClient(path="textbook_db")
    return db_client.get_or_create_collection(name="textbook_collection")

client = genai.Client(api_key=st.session_state.assigned_key)
collection = get_db_collection()

# Restored back to your original Flash model
ai_model = "gemini-3.1-flash-lite" 

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- 3. CORE LOGIC (FLASH OPTIMIZED) ---

def get_context_and_audit(prompt, full_history):
    # Step 1: Distill Query (Remove MCQ options from search to stop poisoning)
    question_stem = prompt.split('\n')[0]
    query_target = re.search(r'([A-H]\d+)', question_stem, re.IGNORECASE)
    search_term = query_target.group(1) if query_target else question_stem

    # Step 2: Retrieve
    r = client.models.embed_content(
        model="gemini-embedding-2-preview", 
        contents=search_term, 
        config={'output_dimensionality': 768}
    )
    # Pulling 8 chunks to make sure we don't miss "Measurable Elements"
    results = collection.query(query_embeddings=[r.embeddings[0].values], n_results=8)
    context = "\n\n---\n\n".join(results['documents'][0])

    # Step 3: Format History cleanly (Flash needs simpler history parsing)
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in full_history[-4:]]) if full_history else "No previous history."
    
    # Step 4: The Strict Audit Prompt (Flash-Lite specific constraints)
    audit_prompt = f"""
    ROLE: Elite, Zero-Tolerance NQAS Compliance Auditor.
    
    PREVIOUS DISCUSSION HISTORY:
    {history_text}
    
    NEW USER QUERY:
    {prompt}
    
    TEXTBOOK CONTEXT:
    {context}
    
    TASK PROTOCOL: 
    1. If the user is asking a NEW MCQ: Provide the 100% correct answer using ONLY the Textbook Context for factual truth. 
    *CRITICAL FOR MCQ*: You must evaluate each option independently. If options a, b, and c are all true based on the text, you MUST select 'All of the above' if it is an option.
    2. If the user is asking a FOLLOW-UP: Answer their specific doubt based strictly on the previous discussion. Do not hallucinate outside rules.
    
    FORMATTING:
    Format strictly with: ✅ CORRECT ANSWER, 📖 TEXTBOOK EVIDENCE, and ⚡ QUICK RATIONALE.
    """
    
    return client.models.generate_content(model=ai_model, contents=audit_prompt).text


# --- 4. STREAMLIT UI EXECUTION ---

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input('Paste MCQ or ask a follow-up...'):
    
    # Render user prompt instantly
    with st.chat_message('user'): 
        st.markdown(prompt)
    
    with st.spinner('Auditing Answer...'): 
        try:
            # Execute audit using Flash-Lite
            response = get_context_and_audit(prompt, st.session_state.messages)
            
            # Display response
            with st.chat_message('assistant'): 
                st.markdown(response)
                
            # Store in session state AFTER successful generation
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
                    
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if "PerDay" in str(e):
                    old_key = st.session_state.assigned_key
                    new_key = random.choice([k for k in keys if k != old_key])
                    st.session_state.assigned_key = new_key
                    
                    # Cycle the key and rerun
                    client = genai.Client(api_key=new_key)
                    st.rerun()
                else:
                    st.warning("⏳ Minute limit hit. Sleeping for 60 seconds...")
                    time.sleep(60)
                    st.rerun()
            else:
                st.error(f"Execution Error: {e}")
