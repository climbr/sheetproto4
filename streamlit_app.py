import streamlit as st
import pandas as pd
import io
from datetime import datetime, date
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber

# Page config
st.set_page_config(
    page_title="Test Case Manager",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .test-case-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .status-passed {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .status-failed {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- NEW ADDITIONS FOR HUGGINGFACE INTEGRATION ---

# Initialize session state for RAG components
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "generated_suggestions" not in st.session_state:
    st.session_state.generated_suggestions = {}

@st.cache_resource
def load_llm_and_embedder():
    """Load the LLM and the embedding model."""
    try:
        llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return llm_pipeline, embedder
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure you have the required libraries installed and an internet connection.")
        return None, None

def process_documents(uploaded_files, embedder):
    """Processes uploaded documents and creates a FAISS vector store."""
    all_text = ""
    for file in uploaded_files:
        if file.type == "text/plain":
            all_text += file.read().decode("utf-8")
        elif file.type == "application/pdf":
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        all_text += page.extract_text() + "\n"
            except Exception as e:
                st.error(f"Error processing PDF file: {e}")
                continue
    
    chunks = [all_text[i:i + 512] for i in range(0, len(all_text), 512)]
    
    if not chunks:
        return None, None
        
    embeddings = embedder.encode(chunks)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, chunks

def generate_test_case_context(test_case_description, vector_store, chunks, llm_pipeline):
    """Retrieves context and generates a response using the LLM."""
    if vector_store is None:
        return "Please upload and process context documents first."
    
    query_embedding = st.session_state.embedder.encode([test_case_description])
    D, I = vector_store.search(query_embedding, 3)
    
    retrieved_docs = [chunks[i] for i in I[0]]
    
    context = " ".join(retrieved_docs)
    
    prompt = f"""
    You are an expert Test Case Manager.
    
    Context: {context}
    
    Based on the context and best practices, provide a detailed analysis for the following test case.
    Suggest improvements, potential edge cases, and ensure the test case is robust.

    Test Case Description: {test_case_description}

    Detailed Analysis and Suggestions:
    """
    
    try:
        response = llm_pipeline(prompt, max_length=512, do_sample=False)
        return response[0]['generated_text']
    except Exception as e:
        return f"Error during text generation: {e}"

# --- END OF NEW ADDITIONS ---

# Main application logic
def main():
    st.markdown("<div class='main-header'><h1>Test Case Manager 🧪</h1></div>", unsafe_allow_html=True)
    
    st.sidebar.title("🛠️ Options")
    
    uploaded_file = st.sidebar.file_uploader("Upload Test Cases CSV", type=["csv"])
    
    st.header("📄 Context & Business Rules")
    uploaded_context_files = st.file_uploader("Upload documents for context (PDF, TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    
    if uploaded_context_files:
        st.session_state.llm_pipeline, st.session_state.embedder = load_llm_and_embedder()
        
        if st.session_state.llm_pipeline and st.session_state.embedder:
            with st.spinner("Processing documents and building knowledge base..."):
                st.session_state.vector_store, st.session_state.chunks = process_documents(uploaded_context_files, st.session_state.embedder)
            
            if st.session_state.vector_store:
                st.success("Documents processed successfully! The AI assistant is ready.")
            else:
                st.warning("Could not process documents. Please check the file contents.")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Test Status' not in df.columns:
            df['Test Status'] = 'Pending'
        
        if 'df' not in st.session_state or not st.session_state.df.equals(df):
            st.session_state.df = df
            st.session_state.show_changes = False
            st.session_state.changes = []
        
        if 'show_changes' not in st.session_state:
            st.session_state.show_changes = False
            st.session_state.changes = []

    if st.session_state.get('df') is not None and not st.session_state.df.empty:
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Filters")
        categories = ["All"] + list(st.session_state.df['Category'].unique())
        selected_category = st.sidebar.selectbox("Filter by Category", categories)
        
        test_statuses = ["All"] + list(st.session_state.df['Test Status'].unique())
        selected_status = st.sidebar.selectbox("Filter by Status", test_statuses)
        
        search_query = st.sidebar.text_input("Search Test Cases")
        
        filtered_df = st.session_state.df.copy()
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['Category'] == selected_category]
        if selected_status != "All":
            filtered_df = filtered_df[filtered_df['Test Status'] == selected_status]
        if search_query:
            filtered_df = filtered_df[filtered_df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]

        st.subheader("📋 Test Cases")
        if filtered_df.empty:
            st.warning("No test cases match the current filters.")
        else:
            display_test_cases(filtered_df)

    elif st.session_state.get('df') is not None and len(st.session_state.df) == 0:
        st.warning("📄 No test cases found in the uploaded CSV file.")
    
    else:
        st.info("👆 Please upload a CSV file to get started.")
        
        with st.expander("📋 Expected CSV Format"):
            st.write("Your CSV should have these columns:")
            sample_data = {
                'ID': ['TC001', 'TC002'],
                'Category': ['Login', 'Navigation'],
                'Test Case': ['Valid Login', 'Menu Navigation'],
                'Test Description': ['Test login with valid credentials', 'Test main menu navigation'],
                'Test Input': ['username: admin, password: 123', 'Click on menu items'],
                'Expected Outcome': ['Successfully logged in', 'Menu items work correctly'],
                'Test Env': ['Chrome, Windows 10', 'Firefox, MacOS'],
                'Observed Outcome': ['', ''],
                'Test Status': ['Pending', 'Pending'],
                'Date of Last Test': ['', ''],
                'Notes': ['', '']
            }
            st.dataframe(pd.DataFrame(sample_data))

def display_test_cases(df):
    for index, row in df.iterrows():
        with st.expander(f"**{row['Test Case']}**"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**ID:** {row['ID']}")
                st.write(f"**Category:** {row['Category']}")
                st.write(f"**Test Env:** {row['Test Env']}")
                
                status_color = "green" if row['Test Status'] == "Passed" else ("red" if row['Test Status'] == "Failed" else "gray")
                st.markdown(f"**Status:** <span style='color:{status_color}'>**{row['Test Status']}**</span>", unsafe_allow_html=True)

                st.markdown("---")
                
                new_status = st.radio("Select New Status", ('Pending', 'Passed', 'Failed'), key=f"status_{index}")
                
                if st.button("Update Status", key=f"update_{index}"):
                    old_status = st.session_state.df.loc[st.session_state.df['ID'] == row['ID'], 'Test Status'].iloc[0]
                    st.session_state.df.loc[st.session_state.df['ID'] == row['ID'], 'Test Status'] = new_status
                    st.session_state.df.loc[st.session_state.df['ID'] == row['ID'], 'Date of Last Test'] = datetime.now().strftime("%Y-%m-%d")
                    st.session_state.df.loc[st.session_state.df['ID'] == row['ID'], 'Observed Outcome'] = ''
                    
                    st.session_state.show_changes
