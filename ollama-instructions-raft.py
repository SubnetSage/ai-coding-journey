import time
import csv
from crewai_tools import tool
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama
from PyPDF2 import PdfReader

# Initialize the LLM model
ollama_llm = Ollama(model="llama3.1")

# Function to read and extract text from each page of the PDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    texts = [page.extract_text() for page in reader.pages if page.extract_text()]
    return texts

# Function to generate instruction data
def generate_instruction_data(texts, llm_model):
    instructions = []
    for i, text in enumerate(texts[:100]):  # Limit to 100 instructions
        # Create instruction for Alpaca style
        instruction = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction += f"### Instruction:\nGenerate a question and answer set based on the following content.\n\n"
        instruction += f"### Input:\n{text}\n\n"
        
        # Get response from LLM
        response = llm_model.generate(prompts=[instruction])
        answer = response.generations[0][0].text.strip()

        # Format the Alpaca style response
        alpaca_response = f"### Response:\n{answer}\n"
        instructions.append([instruction, alpaca_response])
    
    return instructions

# Function to save data to CSV
def save_to_csv(data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Instruction", "Response"])
        for row in data:
            # Use csv.QUOTE_MINIMAL to handle any embedded quotes or commas
            writer.writerow([row[0].replace('\n', ' ').replace('"', '""'), row[1].replace('\n', ' ').replace('"', '""')])

# Track the start time
start_time = time.time()

# Load the document and generate instruction data
texts = extract_text_from_pdf(".pdf")
instruction_data = generate_instruction_data(texts, ollama_llm)

# Save the instruction data to a CSV file
save_to_csv(instruction_data, "instruction_data.csv")

# Track the end time
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60

# Print the elapsed time
print(f"Script took {elapsed_minutes:.2f} minutes to run.")

# Define the instruction generation tool function
@tool("Instruction_Generator")
def instruction_generator_tool(pdf_path: str) -> str:
    """This tool generates instruction data from the content of a given PDF file."""
    texts = extract_text_from_pdf(pdf_path)
    instructions = generate_instruction_data(texts, ollama_llm)
    save_to_csv(instructions, "instruction_data.csv")
    return "Instruction data has been saved to instruction_data.csv."

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

pdf_instruction_role = 'Instruction Generator'
pdf_instruction_goal = 'Generate instruction training data from the provided PDF document.'
pdf_instruction_backstory = f"""
    You are an expert in generating training data for machine learning models. 
    Your task is to read and rewrite each page of the given PDF document into Alpaca style instructions and responses.
    Convert the content into questions and answers that can be used as training data, ensuring clarity and relevance for the context provided.
    Here is a sample of the guide for reference: {instruction_data[:500]}... (truncated)
"""

pdf_instruction_tools = [instruction_generator_tool]

# Create the instruction generation agent using the template
pdf_instruction_agent = create_agent(pdf_instruction_role, pdf_instruction_goal, pdf_instruction_backstory, pdf_instruction_tools, ollama_llm)

# Create the instruction generation task using the template
instruction_task = create_task(pdf_instruction_agent, "Generate instruction training data from the provided PDF document.", "Instruction and question-answer set for each page of the PDF document.")

# Instantiate the Crew with the instruction generation process
crew = Crew(
    agents=[pdf_instruction_agent],
    tasks=[instruction_task],
    verbose=2  # Adjust the logging level as needed (1 or 2)
)

# Get the crew to start working on the instruction generation task
results = crew.kickoff()

# Extract the generated instruction data from the task result
final_instruction_data = results[0].output

# Print the final instruction data
print(final_instruction_data)
