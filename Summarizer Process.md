### High-Level Overview

This script is designed to read a PDF document and summarize each page using an open source language model called Llama3.1. The project is still under development and this script represents the foundational structure. The script initializes the LLM, reads the PDF, generates summaries for each page, and manages the summarization process using a custom agent and task framework.

### Required Libraries

- `time`
- `crewai` (assuming this is a custom or third-party library)
- `langchain_community.llms`
- `PyPDF2`
- **Ollama** and **Python** are essential for this project.

### How to Run

1. **Install the Required Libraries**:
   ```bash
   pip install PyPDF2
   pip install your_crewai_package
   pip install langchain-community
   pip install ollama
   ```

2. **Ensure you have Python and Ollama**:
   - Make sure you have Python installed on your system.
   - Ensure that Ollama is properly configured and available.

3. **Save the Script**: Save the script to a Python file, for example, `pdf_summarizer.py`.

4. **Run the Script**:
   ```bash
   python pdf_summarizer.py
   ```

### Script

```python
import time
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama
from PyPDF2 import PdfReader

# Initialize the LLM model
ollama_llm = Ollama(model="llama3.1")

# Function to read and summarize each page of the PDF guide
def summarize_pdf(file_path, llm_model):
    reader = PdfReader(file_path)
    summaries = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            response = llm_model.generate(prompts=[f"Summarize the following text:\n\n{text}"])
            summary = response.generations[0][0].text
            summaries.append(f"Page {i+1} Summary:\n{summary}\n")
    return "\n".join(summaries)

# Track the start time
start_time = time.time()

# Load the document and summarize each page
pdf_summary = summarize_pdf("net-intro.pdf", ollama_llm)

# Track the end time
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60

# Print the elapsed time
print(f"Script took {elapsed_minutes:.2f} minutes to run.")

# Define the summarization tool function
@tool("PDF_Summarizer")
def pdf_summarizer_tool(pdf_path: str) -> str:
    """This tool summarizes the content of a given PDF file."""
    return summarize_pdf(pdf_path, ollama_llm)

# Template for agent creation
def create_agent(role, goal, backstory, tools, llm, verbose=True, allow_delegation=True):
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools,
        llm=llm,
        verbose=verbose,
        allow_delegation=allow_delegation
    )

# Template for task creation
def create_task(agent, description, expected_output):
    return Task(
        agent=agent,
        description=description,
        expected_output=expected_output
    )

pdf_summarizer_role = 'Credit Expert'
pdf_summarizer_goal = 'Summarize each page of the provided PDF document.'
pdf_summarizer_backstory = f"""
    You are an expert IT instructor specializing in Linux systems administration and open-source technologies. 
    Your task is to read and rewrite each page of the given PDF document to simplify and clarify the key concepts of Linux for a broad audience.
    Break down complex topics into easy-to-understand explanations and provide practical examples from real-world scenarios to ensure clarity and comprehension.
    Here is the guide for reference: {pdf_summary[:500]}... (truncated)
"""

pdf_summarizer_tools = [pdf_summarizer_tool]

# Create the summarization agent using the template
pdf_summarizer_agent = create_agent(pdf_summarizer_role, pdf_summarizer_goal, pdf_summarizer_backstory, pdf_summarizer_tools, ollama_llm)

# Create the summarization task using the template
summarization_task = create_task(pdf_summarizer_agent, "Summarize each page of the provided PDF document.", "A summary of each page of the PDF document.")

# Instantiate the Crew with the summarization process
crew = Crew(
    agents=[pdf_summarizer_agent],
    tasks=[summarization_task],
    verbose=2  # Adjust the logging level as needed (1 or 2)
)

# Get the crew to start working on the summarization task
results = crew.kickoff()

# Extract the summary from the task result
final_summary = results[0].output

# Print the final summary
print(final_summary)
```

This script lays the groundwork for a PDF summarization tool using an LLM, highlighting the initialization, processing, and management of summarization tasks. The project is still under development, and further refinements are expected.
