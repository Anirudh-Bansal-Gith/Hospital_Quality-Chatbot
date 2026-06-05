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
ai_model = "gemini-3.1-flash-lite"

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- 3. THE CORE PIPELINE OVERHAUL ---

def distill_query(user_prompt):
    """
    Strips out MCQ choices to prevent them from poisoning the vector search.
    Extracts strictly the core question stem or standard codes (e.g., 'B6', 'IPHS').
    """
    lines = user_prompt.split('\n')
    question_stem = lines[0]
    
    # Check for direct NQAS standard references (e.g., A1, B6, E3)
    standard_match = re.search(r'([A-H]\d+)', question_stem, re.IGNORECASE)
    if standard_match:
        return standard_match.group(1)
        
    return question_stem


def retrieve_context(clean_query):
    """Fetches a wider window of chunks to ensure complete standard sets are present."""
    r = client.models.embed_content(
        model="gemini-embedding-2-preview", 
        contents=clean_query, 
        config={'output_dimensionality': 768}
    )
    query_vector = r.embeddings[0].values
    
    # Pulling top 8 chunks to fully capture multiple Measurable Elements
    results = collection.query(query_embeddings=[query_vector], n_results=8)
    
    context_list = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        source = f"[BOOK: {meta.get('book', 'N/A')} | PAGE: {meta.get('page_num', 'N/A')}]"
        context_list.append(f"{source}\n{doc}")
        
    return "\n\n---\n\n".join(context_list)


def run_agentic_audit(question, context):
    """
    Executes a two-step generation and verification loop.
    The primary agent solves the question, then an independent auditor verifies the logic.
    """
    
    # Phase 1: Absolute Target Analysis
    generation_prompt = f"""
    ROLE: Ultra-precise NQAS Auditor. Solve this MCQ based strictly on the provided context.
    
    CRITICAL CONSTRAINT: Evaluate every single choice option independently against the text evidence. 
    If options a, b, and c are all separate valid components under this standard framework, 
    you MUST select the collective option (e.g., 'All of the above').

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """
    
    initial_response = client.models.generate_content(model=ai_model, contents=generation_prompt).text
    
    # Phase 2: Adversarial Verification Review
    verification_prompt = f"""
    ROLE: Zero-Tolerance Compliance Verifier.
    Your job is to catch logical errors, hasty conclusions, or missed options in the proposed solution.
    
    PROPOSED SOLVER OUTPUT:
    {initial_response}
    
    RAW REFERENCE CONTEXT:
    {context}
    
    TASK: Validate the proposed choice. Ensure every option mentioned in the answer is verbatim backed by the text.
    If the solver missed a collective option (like 'All of the above'), fix the error completely.
    Output ONLY the final, verified, and polished layout using emojis for visual structure. 
    Format with: ✅ CORRECT ANSWER, 📖 TEXTBOOK EVIDENCE, and ⚡ QUICK RATIONALE.
    """
    
    final_verified_output = client.models.generate_content(model=ai_model, contents=verification_prompt).text
    return final_verified_output


# --- 4. STREAMLIT UI EXECUTION ---

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input('Paste the MCQ here...')

if prompt:
    with st.chat_message('user'):
        st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner('Running Dual-Agent Audit Verification...'): 
        try:
            # Step A: Distill the query to protect the vector search
            clean_search_target = distill_query(prompt)
            
            # Step B: Secure the reference text
            retrieved_text = retrieve_context(clean_search_target)
            
            # Step C: Route through the multi-agent execution ring
            final_answer = run_agentic_audit(prompt, retrieved_text)
            
            with st.chat_message('ai'):
                st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if "PerDay" in str(e):
                    old_key = st.session_state.assigned_key
                    new_key = random.choice([k for k in keys if k != old_key])
                    st.session_state.assigned_key = new_key
                    
                    # Re-initialize the client on key rotation
                    client = genai.Client(api_key=new_key)
                    st.rerun()
                else:
                    st.warning("⏳ Minute limit hit. Sleeping for 60 seconds...")
                    time.sleep(60)
                    st.rerun()
            else:
                st.error(f"Execution Error: {e}")
