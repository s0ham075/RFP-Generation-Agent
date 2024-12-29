from pathlib import Path
from io import BytesIO
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent
from workflow import RFPWorkflow
from tools import generate_tools
from llama_index.llms.azure_openai import AzureOpenAI
import tempfile
import os
import streamlit as st
import streamlit_ace as st_ace
import markdown as md
import asyncio

async def setup_rfp_workflow():
    """Initialize the RFP workflow components"""
    print("Setting up RFP Workflow")
    aoai_api_key = "DY9DKhZV6yPfaEBSOKvALOoTV46Hi4I3Mywr0Sjmdl22XsBg9HeEJQQJ99ALACfhMk5XJ3w3AAAAACOGPCQT"
    aoai_endpoint = "https://subad-m4bjh7zv-swedencentral.cognitiveservices.azure.com/openai/deployments/snapdoc-gpt-4-32k/chat/completions?api-version=2024-08-01-preview"
    aoai_api_version = "2023-05-15"

    llm = AzureOpenAI(
        engine="snapdoc-gpt-4-32k",
        model="gpt-4o-mini",
        api_key=aoai_api_key,
        azure_endpoint=aoai_endpoint,
        api_version=aoai_api_version,
    )

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
        # Save uploaded file
        temp_file_path = Path(temp_dir) / "uploaded_rfp.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        print(f"Uploaded file saved to {temp_file_path}")
        
        # Initialize workflow
        workflow = RFPWorkflow(
            tools=workflow_components['tools'],
            llm=workflow_components['llm'],
            verbose=True,
            output_dir=temp_dir,
            timeout=None
        )
        print("Workflow initialized")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run workflow
        handler = workflow.run(rfp_template_path=str(temp_file_path))
        
        markdown_content = []
        qna_content = []
        async for event in handler.stream_events():
            print(f"Event received type: {type(event)}")
            if isinstance(event, InputRequiredEvent):
                st.session_state.current_qna = qna_content
                return None, qna_content  # Return early to handle Q&A editing

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
        
        # Get final response
        response = await handler
        progress_bar.progress(1.0)
        print("Workflow completed")
        
        final_markdown = str(response)
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

def handle_qna_submission():
    """Handle Q&A submission and update workflow"""
    if st.session_state.edited_qna is not None:
        final_qna = "\n\n".join(f"Q: {row['question']}\nA: {row['answer']}" 
                               for _, row in st.session_state.edited_qna.iterrows())
        st.session_state.workflow_handler.ctx.send_event(HumanResponseEvent(response=final_qna))
        st.session_state.qna_submitted = True

async def main():
    print("Initializing Streamlit application")
    st.set_page_config(layout="wide")
    st.title("RFP Response Generator")

    # Initialize session state
    if 'workflow_components' not in st.session_state:
        st.session_state.workflow_components = await setup_rfp_workflow()
    if 'show_editor' not in st.session_state:
        st.session_state.show_editor = False
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'current_qna' not in st.session_state:
        st.session_state.current_qna = None
    if 'qna_submitted' not in st.session_state:
        st.session_state.qna_submitted = False
    if 'workflow_handler' not in st.session_state:
        st.session_state.workflow_handler = None

    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload RFP")
        uploaded_file = st.file_uploader("Choose an RFP file", type="pdf")

    if uploaded_file is not None and st.session_state.workflow_components is not None:
        try:
            # Only process if we haven't already or if it's a new file
            if st.session_state.processed_data is None or uploaded_file != st.session_state.last_uploaded_file:
                print("Processing new RFP file")
                with st.spinner("Processing RFP..."):
                    markdown_text, qna_content = await process_rfp_file(
                        uploaded_file, 
                        st.session_state.workflow_components
                    )
                    st.session_state.processed_data = markdown_text
                    st.session_state.current_qna = qna_content
                    st.session_state.last_uploaded_file = uploaded_file

            # Handle Q&A editing if needed
            if st.session_state.current_qna is not None and not st.session_state.qna_submitted:
                st.subheader("Edit Extracted Q&A Content")
                qna_df = pd.DataFrame(st.session_state.current_qna)
                st.session_state.edited_qna = st.data_editor(
                    qna_df, 
                    num_rows="dynamic",
                    use_container_width=True,
                    key="qna_editor"
                )
                
                if st.button("Submit Q&A", key="submit_qna"):
                    handle_qna_submission()

            # Display final content
            if st.session_state.processed_data is not None:
                if st.button("Toggle Editor/Renderer"):
                    st.session_state.show_editor = not st.session_state.show_editor

                if st.session_state.show_editor:
                    st.subheader("Edit Response Content")
                    edited_markdown_text = st_ace.st_ace(
                        value=st.session_state.processed_data,
                        language='markdown',
                        theme='monokai',
                        key='markdown_editor',
                    )
                    st.session_state.processed_data = edited_markdown_text
                else:
                    st.subheader("Rendered Response Content")
                    st.markdown(st.session_state.processed_data, unsafe_allow_html=True)

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Markdown",
                        data=st.session_state.processed_data,
                        file_name="rfp_response.md",
                        mime="text/markdown"
                    )

                with col2:
                    pdf = convert_markdown_to_pdf(st.session_state.processed_data)
                    st.download_button(
                        label="Download as PDF",
                        data=pdf,
                        file_name="rfp_response.pdf",
                        mime="application/pdf"
                    )

        except Exception as e:
            st.error(f"Error processing RFP: {str(e)}")
            print(f"Error encountered: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())