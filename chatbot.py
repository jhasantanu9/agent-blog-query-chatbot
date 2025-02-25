import streamlit as st
import pandas as pd
from io import StringIO
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.prompts import PromptTemplate

# Sample data file path
SAMPLE_DATA = "blog_dataset.csv"

# Page configuration
st.set_page_config(
    page_title="AI Blog Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for storing DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar with description and data selection
with st.sidebar:
    st.title("📚 About")
    st.markdown("""
    ### Welcome to AI Blog Assistant!
    
    This tool helps you analyze and query your blog content using AI. Simply:
    1. Choose between sample data or upload your own
    2. Wait for the embeddings to generate
    3. Ask questions about your blogs
    
    #### CSV Requirements:
    - Must contain columns: 'Title', 'Keyword', 'Content'
    - File should be in CSV format
    
    #### Features:
    - Natural language querying
    - Context-aware responses
    - Basic analytics
    - Source attribution
    - Powered by Google's Gemini AI
    """)
    
    st.divider()
    
    # Data source selection
    st.subheader("📁 Choose Data Source")
    data_source = st.radio("Select data source:", ["Upload Your Own CSV", "Use Sample Data"])
    
    if data_source == "Upload Your Own CSV":
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
        
        if uploaded_file:
            file_buffer = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(file_buffer)
            
            required_columns = {"Title", "Keyword", "Content"}
            if not required_columns.issubset(set(df.columns)):
                st.error("❌ CSV must contain 'Title', 'Keyword', and 'Content' columns.")
            else:
                # Clean and validate data
                df = df.fillna("")
                for col in ["Title", "Keyword", "Content"]:
                    df[col] = df[col].astype(str)
                
                st.session_state.df = df
                st.success("✅ CSV uploaded successfully!")
    else:
        try:
            # Load sample data from CSV file
            df = pd.read_csv(SAMPLE_DATA)
            # Clean and validate data
            df = df.fillna("")
            for col in ["Title", "Keyword", "Content"]:
                if col not in df.columns:
                    st.error(f"❌ Sample dataset missing required column: {col}")
                    st.session_state.df = None
                    break
                df[col] = df[col].astype(str)
            else:
                st.session_state.df = df
                st.success("✅ Sample data loaded successfully!")
        except FileNotFoundError:
            st.error(f"❌ Sample dataset file '{SAMPLE_DATA}' not found!")
            st.session_state.df = None
        except Exception as e:
            st.error(f"❌ Error loading sample dataset: {str(e)}")
            st.session_state.df = None

    st.markdown("---")

# Main content area
st.title("🤖 AI Blog Assistant")

# Set the API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

def get_analytics_response(query, df):
    """Handle analytics-related queries about the dataset"""
    query_lower = query.lower()
    
    if "how many" in query_lower and "blog" in query_lower:
        return f"The dataset contains {len(df)} blog posts."
    
    if "keywords" in query_lower:
        keywords = df['Keyword'].value_counts()
        return f"Top keywords in the blogs:\n{keywords.head().to_string()}"
        
    return None

def get_blog_content_response(query, df):
    """Get relevant blog content for specific queries"""
    query_words = set(query.lower().split())
    relevant_blogs = []
    
    for _, row in df.iterrows():
        content_words = set(row['Content'].lower().split())
        if query_words & content_words:
            relevant_blogs.append(row['Content'])
    
    return relevant_blogs[:3] if relevant_blogs else None

# Initialize components only if data is available
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Load Sentence Transformer for Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="local_model")
    
    # Create custom prompt template
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say you don't know. Don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Please provide a comprehensive answer based on the context. If the question is about specific events, 
    policies, or actions, cite specific examples from the blogs when available.
    
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Generate embeddings if not already done
    if 'vector_store' not in st.session_state:
        with st.spinner("🔄 Generating embeddings... (This may take time for large files)"):
            # Convert all columns to string first
            df["Title"] = df["Title"].astype(str)
            df["Keyword"] = df["Keyword"].astype(str)
            df["Content"] = df["Content"].astype(str)
            
            # Then combine them
            df["combined_text"] = df["Title"] + " " + df["Keyword"] + " " + df["Content"]
            texts = df["combined_text"].tolist()
            st.session_state.vector_store = FAISS.from_texts(texts, embedding_model)

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.3,
        top_p=0.8,
        convert_system_message_to_human=True
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    # Chat interface
    st.markdown("### 💬 Ask me anything about your blogs")
    query = st.text_input("Type your question here:", key="query_input")
    
    if query:
        with st.spinner("🔍 Searching for answer..."):
            try:
                # First check if it's an analytics question
                analytics_response = get_analytics_response(query, df)
                if analytics_response:
                    st.markdown("### 📊 Analytics Response")
                    st.write(analytics_response)
                else:
                    # Try regular QA chain
                    response = qa_chain.run(query)
                    
                    # Display answer in a nice container
                    st.markdown("### 📝 Answer")
                    st.write(response)
                    
                    # Display sources in an expander
                    with st.expander("📚 View Sources"):
                        top_matches = st.session_state.vector_store.similarity_search(query, k=3)
                        for i, match in enumerate(top_matches, 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(f"```\n{match.page_content[:500]}...\n```")
                            st.divider()
                            
                    # Additional relevant blogs
                    relevant_blogs = get_blog_content_response(query, df)
                    if relevant_blogs:
                        with st.expander("📑 Additional Relevant Blog Content"):
                            for i, blog in enumerate(relevant_blogs, 1):
                                st.markdown(f"**Blog {i}:**")
                                st.markdown(f"```\n{blog[:500]}...\n```")
                                st.divider()
                        
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
else:
    st.info("👈 Please select a data source in the sidebar to get started!")
