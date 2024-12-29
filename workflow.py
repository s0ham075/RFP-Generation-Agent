from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent
import PyPDF2
import re
from pathlib import Path
from typing import List
from llama_index.core.llms import LLM
from typing import Optional
from pydantic import BaseModel
from llama_index.core.schema import Document
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
import json
import os

data_out_dir = "data_out_rfp"
g_output_template = ""

# this is the research agent's system prompt, tasked with answering a specific question
AGENT_SYSTEM_PROMPT = """\
You are a research agent tasked with filling out a specific form key/question with the appropriate value, given a bank of context.
You are given a specific form key/question. Think step-by-step and use the existing set of tools to help answer the question.

You MUST always use at least one tool to answer each question. Only after you've determined that existing tools do not \
answer the question should you try to reason from first principles and prior knowledge to answer the question.

You MUST try to answer the question instead of only saying 'I dont know'.

"""

# This is the prompt tasked with extracting information from an RFP file.
EXTRACT_KEYS_PROMPT = """\
You are provided an entire RFP document, or a large subsection from it. 

We wish to generate a response to the RFP in a way that adheres to the instructions within the RFP, \
including the specific sections that an RFP response should contain, and the content that would need to go \
into each section.

Your task is to extract out a list of "questions", where each question corresponds to a specific section that is required in the RFP response.
Put another way, after we extract out the questions we will go through each question and answer each one \
with our downstream research assistant, and the combined
question:answer pairs will constitute the full RFP response.

You must TRY to extract out questions that can be answered by the provided knowledge base. We provide the list of file metadata below. 

Additional requirements:
- Try to make the questions SPECIFIC given your knowledge of the RFP and the knowledge base. Instead of asking a question like \
"How do we ensure security" ask a question that actually addresses a security requirement in the RFP and can be addressed by the knowledge base.
- Make sure the questions are comprehensive and addresses all the RFP requirements.
- Make sure each question is descriptive - this gives our downstream assistant context to fill out the value for that question 
- Extract out all the questions as a list of strings.


Knowledge Base Files:
{file_metadata}

RFP Full Template:
{rfp_text}

"""

# this is the prompt that generates the final RFP response given the original template text and question-answer pairs.
GENERATE_OUTPUT_PROMPT = """\
You are an expert analyst.
Your task is to generate an RFP response according to the given RFP and question/answer pairs.

You are given the following RFP and qa pairs:

<rfp_document>
{output_template}
</rfp_document>

<question_answer_pairs>
{answers}
</question_answer_pairs>

Not every question has an appropriate answer. This is because the agent tasked with answering the question did not have the right context to answer it.
If this is the case, you MUST come up with an answer that is reasonable. You CANNOT say that you are unsure in any area of the RFP response.


Please generate the output according to the template and the answers, in markdown format.
Directly output the generated markdown content, do not add any additional text, such as "```markdown" or "Here is the output:".
Follow the original format of the template as closely as possible, and fill in the answers into the appropriate sections.
"""


class OutputQuestions(BaseModel):
    """List of keys that make up the sections of the RFP response."""

    questions: List[str]


class OutputTemplateEvent(Event):
    docs: List[Document]


class QuestionsExtractedEvent(Event):
    questions: List[str]


class HandleQuestionEvent(Event):
    question: str


class QuestionAnsweredEvent(Event):
    question: str
    answer: str


class CollectedAnswersEvent(Event):
    combined_answers: str

class CheckCollectedAnswersEvent(Event):
    combined_answers: str

class LogEvent(Event):
    msg: str
    delta: bool = False
    question: Optional[str] = None
    answer: Optional[str] = None

class CustomPDFParser:
    """Custom PDF parser with proper character encoding handling."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing problematic characters and normalizing whitespace."""
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text.replace('\uf0b7', '-').replace('\u2022', '-').replace('\u2023', '-').replace('\u2043', '-'))
    
    @staticmethod
    async def parse_pdf(file_path: str) -> List[Document]:
        documents = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                cleaned_text = CustomPDFParser.clean_text(text)
                if cleaned_text.strip():
                    documents.append(Document(
                        text=cleaned_text,
                        metadata={
                            "page_number": page_num + 1,
                            "file_path": file_path,
                            "total_pages": len(pdf_reader.pages)
                        }
                    ))
        return documents


class RFPWorkflow(Workflow):
    """RFP workflow."""

    def __init__(
        self,
        tools,
        llm: LLM | None = None,
        similarity_top_k: int = 20,
        output_dir: str = data_out_dir,
        agent_system_prompt: str = AGENT_SYSTEM_PROMPT,
        generate_output_prompt: str = GENERATE_OUTPUT_PROMPT,
        extract_keys_prompt: str = EXTRACT_KEYS_PROMPT,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tools = tools
        self.parser = CustomPDFParser()

        self.llm = llm 
        self.similarity_top_k = similarity_top_k

        self.output_dir = output_dir

        self.agent_system_prompt = agent_system_prompt
        self.extract_keys_prompt = extract_keys_prompt

        out_path = Path(self.output_dir) / "workflow_output"
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
            os.chmod(str(out_path), 0o0777)

        self.generate_output_prompt = PromptTemplate(generate_output_prompt)

    @step
    async def parse_output_template(
        self, ctx: Context, ev: StartEvent
    ) -> OutputTemplateEvent:
        print("Step: Parse Output Template")
        docs = await self.parser.parse_pdf(ev.rfp_template_path)
        print(f"Parsed {len(docs)} documents from the template.")
        await ctx.set("output_template", docs)
        return OutputTemplateEvent(docs=docs)

    @step
    async def extract_questions(
        self, ctx: Context, ev: OutputTemplateEvent
    ) -> HandleQuestionEvent:
        ctx.write_event_to_stream(LogEvent(msg=f"Extracting questions"))
        docs = ev.docs
        all_text = "\n\n".join([d.get_content(metadata_mode="all") for d in docs])

        response = await self.llm.acomplete(
            self.extract_keys_prompt.format(
                file_metadata="\n\n".join([
                    f"Name:{t.metadata.name}\nDescription:{t.metadata.description}"
                    for t in self.tools
                ]),
                rfp_text=all_text
            )
        )
        response_text = response.text.strip()
        questions = [line.strip('- ').strip() for line in response_text.split('\n') if line.strip()]
        print(f"Extracted {len(questions)} questions.")
        await ctx.set("num_to_collect", len(questions))

        for question in questions:
            ctx.send_event(HandleQuestionEvent(question=question))

        return None

    @step
    async def handle_question(
        self, ctx: Context, ev: HandleQuestionEvent
    ) -> QuestionAnsweredEvent:
        #print(f"Step: Handle Question - {ev.question}")
        research_agent = FunctionCallingAgentWorker.from_tools(
            self.tools, llm=self.llm, verbose=False, system_prompt=self.agent_system_prompt
        ).as_agent()

        response = await research_agent.aquery(ev.question)
        #print(f"Answered Question: {ev.question}")
        ctx.write_event_to_stream(LogEvent(msg="Finding answers",question= str(ev.question),answer= str(response)))

        return QuestionAnsweredEvent(question=ev.question, answer=str(response))

    @step
    async def combine_answers(
        self, ctx: Context, ev: QuestionAnsweredEvent
    ) -> CheckCollectedAnswersEvent:
        num_to_collect = await ctx.get("num_to_collect")
        results = ctx.collect_events(ev, [QuestionAnsweredEvent] * num_to_collect)
        if results is None:
            return None

        combined_answers = "\n".join([result.model_dump_json() for result in results])
        #print(f"Combined all answers. - \n {combined_answers}")
        with open(
            f"{self.output_dir}/workflow_output/combined_answers.jsonl", "w"
        ) as f:
            f.write(combined_answers)

        return CheckCollectedAnswersEvent(combined_answers=combined_answers)
    
    @step
    async def check_collected_answers(self, ctx: Context, ev: CheckCollectedAnswersEvent) -> InputRequiredEvent:
        g_output_template = await ctx.get("output_template")
        return InputRequiredEvent(prefix=ev.combined_answers)
    
    @step
    async def human_response(self, ctx: Context, ev: HumanResponseEvent) -> CollectedAnswersEvent:
        return CollectedAnswersEvent(combined_answers=ev.response)
    
    @step
    async def generate_output(
        self, ctx: Context, ev: CollectedAnswersEvent
    ) -> StopEvent:
        print("Step: Generate Final Output")
        ctx.write_event_to_stream(LogEvent(msg=f"GENERATING FINAL OUTPUT"))
        output_template = await ctx.get("output_template")
        output_template = "\n".join(
            [doc.get_content("none") for doc in output_template]
        )

        resp = await self.llm.astream(
            self.generate_output_prompt,
            output_template=output_template,
            answers=ev.combined_answers,
        )

        final_output = ""
        async for r in resp:
            final_output += r

        with open(f"{self.output_dir}/workflow_output/final_output.md", "w") as f:
            f.write(final_output)

        #print(f"Final output generated - \n {final_output}")
        return StopEvent(result=final_output)


async def generate_final_output(qna:str, llm):
        print("Step: Generate Final Output")
       
        output_template = g_output_template
        output_template = "\n".join(
            [doc.get_content("none") for doc in output_template]
        )

        resp = await llm.astream(
            GENERATE_OUTPUT_PROMPT,
            output_template=output_template,
            answers=qna,
        )

        final_output = ""
        async for r in resp:
            final_output += r

        return final_output
