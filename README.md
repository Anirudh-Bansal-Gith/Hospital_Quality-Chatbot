# 🏥 AI Quality Assist: Hospital Operations Guide

**Bridging the gap between healthcare management theory and clinical reality.**

AI Quality Assist is a specialized RAG (Retrieval-Augmented Generation) platform designed for hospital administrators, quality managers, and clinical leads. It transforms dense textbook data into actionable, bedside-ready implementation plans using the power of **Gemini 2.5 Flash-lite**.

---

### 🌐 Try It Now
The application is available for free use at:  
👉 **[https://hospital-quality-chatbot.streamlit.app/](https://hospital-quality-chatbot.streamlit.app/)**

---

### 🌟 Key Capabilities

* **📚 Textbook-Grounded Insights**: Uses a vector database (ChromaDB) to ensure every piece of advice is rooted in verified healthcare management literature.
* **⚡ High-Performance AI**: Powered by Google's Gemini 2.5 Flash-lite for rapid, intelligent, and context-aware responses.
* **🛠️ Implementation-Focused**: Moves beyond "what" to "how," providing step-by-step action plans, "Pro-Tips," and "Insiders" management advice.
* **⚖️ Built-in Reliability**: Features a custom API load-balancing system to ensure 99.9% uptime by rotating through multiple processing keys.
* **🎨 Optimized UX**: A custom-engineered Streamlit interface designed for professional environments, featuring a persistent header and clean chat flow.

---

### 🧠 The Consultant Logic

The AI doesn't just "chat"—it follows a rigorous **Senior Hospital Operations Consultant** persona:

1.  **Analyze**: It embeds the user query to find the top 5 most relevant textbook snippets.
2.  **Clean**: It automatically strips OCR "garbage" and formatting errors from the source data.
3.  **Synthesize**: It blends textbook theory with global healthcare standards (WHO, NABH).
4.  **Execute**: It delivers a visually structured, emoji-rich guide that is easy to read during a busy hospital shift.

---

### 📊 System Architecture

* **Core Model**: Gemini 2.5 Flash-lite
* **Embeddings**: Gemini Embedding 2 Preview
* **Vector Engine**: ChromaDB
* **Framework**: Streamlit (Custom CSS/HTML)

---

### 📁 Project Structure

* `app.py`: Main Streamlit application and UI logic.
* `textbook_db/`: Local ChromaDB vector store (Private/Excluded).
* `.streamlit/secrets.toml`: API key configuration (Private/Excluded).
* `requirements.txt`: Python dependencies.
* `logo.jpeg`: Application branding assets.

---

### 📝 Privacy & Standards

This tool is designed to work with local vector stores to maintain data sovereignty and adheres to professional hospital management formatting standards.

---

© 2026 Hospital Quality Guide Team
