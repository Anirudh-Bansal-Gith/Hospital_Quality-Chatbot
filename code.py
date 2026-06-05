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
ai_model = "gemini-2.5-flash" 

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- 3. TWO-PASS AUDIT ENGINE ---

def get_context_and_audit(prompt, full_history):
    # Step A: Distill Query (Remove MCQ options from search to avoid embedding poisoning)
    lines = [line.strip() for line in prompt.split('\n') if line.strip()]
    question_stem = lines[0] if lines else prompt
    
    search_term = question_stem
    for line in lines[:3]:
        match = re.search(r'([A-H]\d+)', line, re.IGNORECASE)
        if match:
            search_term = match.group(1)
            break

    # Step B: Vector Database Query
    r = client.models.embed_content(
        model="gemini-embedding-2-preview", 
        contents=search_term, 
        config={'output_dimensionality': 768}
    )
    results = collection.query(query_embeddings=[r.embeddings[0].values], n_results=8)
    raw_context = "\n\n---\n\n".join(results['documents'][0])

    # Step C: Format conversation history window
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in full_history[-3:]]) if full_history else "No history."

    # --- PASS 1: FACT ISOLATION ---
    extraction_prompt = f"""
    You are a clinical data extraction bot. Read the NQAS reference text below and pull out EVERY explicit rule, numeric limit, time frame, or measurable element description related strictly to: "{search_term}".
    Do not look at any multiple choice options. Do not extrapolate. Give me plain raw facts.
    
    TEXTBOOK CONTEXT:
    {raw_context}
    """
    verified_facts = client.models.generate_content(model=ai_model, contents=extraction_prompt).text

    # --- PASS 2: ADVERSARIAL AUDIT ---
    audit_prompt = f"""
    ROLE: Elite, Zero-Tolerance NQAS Compliance Auditor.
    
    PREVIOUS DISCUSSION HISTORY:
    {history_text}
    
    TARGET QUESTION & OPTIONS:
    {prompt}
    
    VERIFIED TEXTBOOK FACTS:
    {verified_facts}
    
    AUDIT PROTOCOL:
    1. Evaluate choice options (a), (b), (c), and (d) completely INDEPENDENTLY against the VERIFIED TEXTBOOK FACTS.
    2. Check for collective options: If options (a), (b), and (c) are all separately accurate components found in the facts, you are strictly REQUIRED to select the collective answer (e.g., 'All of the above').
    3. Do not jump to the first positive keyword. Validate every option systematically before declaring the choice.
    
    FORMATTING:
    Format strictly with: ✅ CORRECT ANSWER, 📖 TEXTBOOK EVIDENCE, and ⚡ QUICK RATIONALE. Use emojis cleanly.
    """
    
    return client.models.generate_content(model=ai_model, contents=audit_prompt).text


# --- 4. STREAMLIT UI EXECUTION ---

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input('Paste MCQ or ask a follow-up...'):
    
    with st.chat_message('user'): 
        st.markdown(prompt)
    
    with st.spinner('Auditing Answer...'): 
        try:
            response = get_context_and_audit(prompt, st.session_state.messages)
            
            with st.chat_message('assistant'): 
                st.markdown(response)
                
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
                    
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if "PerDay" in str(e):
                    old_key = st.session_state.assigned_key
                    new_key = random.choice([k for k in keys if k != old_key])
                    st.session_state.assigned_key = new_key
                    
                    client = genai.Client(api_key=new_key)
                    st.rerun()
                else:
                    st.warning("⏳ Minute limit hit. Sleeping for 60 seconds...")
                    time.sleep(60)
                    st.rerun()
            else:
                st.error(f"Execution Error: {e}")
