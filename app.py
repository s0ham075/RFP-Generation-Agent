from pathlib import Path
from dotenv import load_dotenv
from io import BytesIO
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent
from workflow import RFPWorkflow, generate_final_output
from tools import generate_tools
from llama_index.llms.azure_openai import AzureOpenAI
import tempfile
import os
import streamlit as st
import streamlit_ace as st_ace
import markdown as md
import asyncio

load_dotenv()

def initialize_session_state():
    """Initialize all required session state variables"""
    if 'semaphore' not in st.session_state:
        st.session_state.semaphore = 0
    if 'qna_content' not in st.session_state:
        st.session_state.qna_content = None
    if 'workflow_components' not in st.session_state:
        st.session_state.workflow_components = None
    if 'show_editor' not in st.session_state:
        st.session_state.show_editor = False

def handle_qna_submission(handler, qna_df):
    """Handle the submission of Q&A content"""
    final_qna = "\n\n".join(
        f"Q: {row['question']}\nA: {row['answer']}" for _, row in qna_df.iterrows()
    )
    st.session_state.qna_content = final_qna
    if handler:
        handler.ctx.send_event(HumanResponseEvent(response=final_qna))

async def setup_rfp_workflow():
    """Initialize the RFP workflow components"""
    print("Setting up RFP Workflow")
    
    aoai_api_key = os.getenv("AOAI_API_KEY")
    aoai_endpoint = os.getenv("AOAI_ENDPOINT")
    aoai_api_version = os.getenv("AOAI_API_VERSION")

    llm = AzureOpenAI(
        engine="snapdoc-gpt-4-32k",
        model="gpt-4o-mini",
        api_key=aoai_api_key,
        azure_endpoint=aoai_endpoint,
        api_version=aoai_api_version,
    )

    st.session_state.llm = llm
    tools = generate_tools()
    print("Tools and LLM setup complete")
    return {
        'llm': llm,
        'tools': tools,
    }

async def process_rfp_file(uploaded_file, workflow_components):
    """Process the uploaded RFP file using the workflow"""
    print("Starting RFP file processing")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "uploaded_rfp.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        print(f"Uploaded file saved to {temp_file_path}")
        
        workflow = RFPWorkflow(
            tools=workflow_components['tools'],
            llm=workflow_components['llm'],
            verbose=True,
            output_dir=temp_dir,
            timeout=None
        )
        print("Workflow initialized")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        handler = workflow.run(rfp_template_path=str(temp_file_path))
        
        markdown_content = []
        qna_content = []
        
        async for event in handler.stream_events():
            print(f"Event received type: {type(event)}")
            if isinstance(event, InputRequiredEvent):
                with st.form("my_form"):
                    st.subheader("Edit Extracted Q&A Content")
                    st.session_state.semaphore = 1
                    qna_df = pd.DataFrame(qna_content)
                    edited_qna_df = st.data_editor(qna_df, num_rows="dynamic", use_container_width=True)
                    submit = st.form_submit_button(
                        label="Submit",
                        on_click=handle_qna_submission,
                        args=(handler, edited_qna_df)
                    )
                
            if hasattr(event, "delta") and hasattr(event, "msg"):
                if event.delta:
                    status_text.text_area(
                        "Processing Output:",
                        value=event.msg,
                        height=100
                    )
                    markdown_content.append(event.msg)
                else:
                    status_text.text(event.msg)
                
                if "Extracting questions" in event.msg:
                    progress_bar.progress(0.25)
                elif "Finding answers" in event.msg:
                    progress_bar.progress(0.50)
                    qna_content.append({"question": event.question, "answer": event.answer})
                elif "GENERATING FINAL OUTPUT" in event.msg:
                    progress_bar.progress(0.75)
        
        response = await handler
        progress_bar.progress(1.0)
        print("Workflow completed, progress: 100%")
        
        final_markdown = str(response)
        print("Final markdown generated from response")
            
        return final_markdown, qna_content

def convert_markdown_to_pdf(markdown_text):
    """Convert markdown to PDF using reportlab"""
    print("Converting markdown to PDF")
    html_text = md.markdown(markdown_text)
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(html_text, styles["Normal"])]
    doc.build(story)
    buffer.seek(0)
    print("PDF conversion complete")
    return buffer

async def main():
    initialize_session_state()
    
    print("Initializing Streamlit application")
    st.set_page_config(layout="wide")
    st.title("RFP Response Generator")

    if st.session_state.workflow_components is None and st.session_state.semaphore != 1:
        st.session_state.workflow_components = await setup_rfp_workflow()

    with st.sidebar:
        st.header("Upload RFP")
        uploaded_file = st.file_uploader("Choose an RFP file", type="pdf")

    if uploaded_file is not None and st.session_state.workflow_components is not None:
        print("File uploaded, starting processing")
        with st.spinner("Processing RFP..."):
            if st.session_state.semaphore != 1:
                markdown_text, qna_content = await process_rfp_file(
                    uploaded_file, 
                    st.session_state.workflow_components
                )
            print("Processing complete")
        
        with st.spinner("Generating final output..."):
            if st.session_state.qna_content is not None and st.session.semaphore != 1:
                markdown_text = await generate_final_output(
                    st.session_state.qna_content,
                    st.session_state.llm
                )
        
        if st.button("Toggle Editor/Renderer"):
            st.session_state.show_editor = not st.session_state.show_editor

        if st.session_state.show_editor:
            st.subheader("Edit Response Content")
            edited_markdown_text = st_ace.st_ace(
                value=markdown_text,
                language='markdown',
                theme='monokai',
                key='markdown_editor',
            )
            print("Markdown editor loaded")
        else:
            st.subheader("Rendered Response Content")
            st.markdown(markdown_text, unsafe_allow_html=True)
            print("Markdown rendered in UI")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Markdown",
                data=markdown_text,
                file_name="rfp_response.md",
                mime="text/markdown"
            )
            print("Markdown download button added")

        with col2:
            pdf = convert_markdown_to_pdf(markdown_text)
            st.download_button(
                label="Download as PDF",
                data=pdf,
                file_name="rfp_response.pdf",
                mime="application/pdf"
            )
            print("PDF download button added")

if __name__ == "__main__":
    asyncio.run(main())