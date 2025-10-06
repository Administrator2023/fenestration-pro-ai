import streamlit as st
import os

st.set_page_config(
    page_title="Fenestration Pro AI",
    page_icon="🏗️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 2rem;
    border-radius: 1rem;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.user-message {
    background-color: #e0e7ff;
    margin-left: 2rem;
}
.assistant-message {
    background-color: #f3f4f6;
    margin-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏗️ Fenestration Pro AI</h1>
    <p>Intelligent Document Q&A System - Cloud Edition</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get API key from secrets or user input
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    
    if not api_key:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your key from platform.openai.com"
        )
    else:
        st.success("✅ API Key loaded from secrets")
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📎 Upload PDF",
        type=['pdf'],
        help="Upload a PDF to analyze"
    )
    
    if uploaded_file:
        st.success(f"📄 {uploaded_file.name} uploaded!")
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    st.metric("Total Queries", len(st.session_state.get('messages', [])) // 2)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Welcome to Fenestration Pro AI! I'm here to help answer your questions about windows, doors, and building envelope systems. Upload a PDF or ask me anything!"
    })

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-message user-message">👤 {message["content"]}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message assistant-message">🤖 {message["content"]}</div>', 
                   unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about fenestration, windows, doors, or upload a PDF..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("🤔 Thinking..."):
        if api_key:
            try:
                import openai
                openai.api_key = api_key
                
                # Include context about uploaded file if present
                context = ""
                if uploaded_file:
                    context = f"\n\nNote: The user has uploaded a PDF file: {uploaded_file.name}"
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert in fenestration, windows, doors, glazing systems, and building envelope. Provide detailed, technical answers."},
                        {"role": "user", "content": prompt + context}
                    ],
                    temperature=0.7,
                    max_tokens=800
                )
                
                assistant_response = response.choices[0].message['content']
            except Exception as e:
                assistant_response = f"Error: {str(e)}. Please check your API key."
        else:
            assistant_response = "Please add your OpenAI API key in the sidebar to enable AI responses."
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Fenestration Pro AI - Deployed with ❤️ | 
    <a href='https://github.com/administrator2023/fenestration-pro-ai' style='color: #3b82f6;'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)
