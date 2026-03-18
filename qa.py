import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class DocumentQABot:
    def __init__(self):
        # Initialize free, local embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize Qwen via DashScope compatible endpoint
        self.llm = ChatOpenAI(
            api_key="sk-3283c6eecab24834b572ff77e168904f",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-max", 
            temperature=0.5,
            max_tokens=1024
        )
        self.rag_chain = None

    def load_and_process_file(self, file_path):
        """Loads a document, chunks it, and sets up the RAG chain."""
        try:
            # 1. Load based on extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                docs = PyPDFLoader(file_path).load()
            elif ext in [".docx", ".doc"]:
                docs = Docx2txtLoader(file_path).load()
            elif ext == ".csv":
                docs = CSVLoader(file_path).load()
            elif ext == ".txt":
                docs = TextLoader(file_path).load()
            else:
                return "Error: Unsupported file type. Please upload a PDF, DOCX, CSV, or TXT."

            # 2. Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)

            # 3. Create Vector Store and Retriever
            vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # 4. Set up the Prompt and Chain
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. "
                "\n\n"
                "{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            return "✅ Document processed successfully! You can now ask questions."
        
        except Exception as e:
            return f"❌ Error processing document: {str(e)}"

    def answer_question(self, question):
        """Invokes the RAG chain to answer the user's question."""
        if not self.rag_chain:
            return "Please upload and process a document first."
        if not question.strip():
            return "Please ask a valid question."
        
        try:
            response = self.rag_chain.invoke({"input": question})
            return response["answer"]
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# --- Gradio Interface Setup ---

# Instantiate our bot
qa_bot = DocumentQABot()

# Wrapper function for the Gradio file upload component
def handle_file_upload(file):
    if file is None:
        return "No file uploaded."
    # Gradio passes a temporary file object. 'file.name' gives the absolute path to it.
    return qa_bot.load_and_process_file(file.name)

# Wrapper function for the Gradio text input component
def handle_question(question):
    return qa_bot.answer_question(question)

# Build the UI layout
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🤖 Qwen Document QA Bot")
    gr.Markdown("Upload a document (.pdf, .docx, .csv, .txt) and ask questions about its content.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="1. Upload Document")
            status_output = gr.Textbox(label="Status", interactive=False)
            
            # Trigger the processing when a file is uploaded
            file_input.upload(
                fn=handle_file_upload, 
                inputs=[file_input], 
                outputs=[status_output]
            )

        with gr.Column(scale=2):
            question_input = gr.Textbox(label="2. Ask a Question", placeholder="What is this document about?")
            ask_button = gr.Button("Submit Question", variant="primary")
            answer_output = gr.Textbox(label="Answer", lines=10, interactive=False)
            
            # Trigger the answer generation when the button is clicked
            ask_button.click(
                fn=handle_question, 
                inputs=[question_input], 
                outputs=[answer_output]
            )

# Launch the app
if __name__ == "__main__":
    interface.launch()