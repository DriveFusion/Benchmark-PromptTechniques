import os
import time
import asyncio  # Required for running async functions
from groq import Groq
from tabulate import tabulate  # You can install this using pip install tabulate

# Import necessary components for the workflow
from llama_index.core.workflow import (
    Workflow,
    Context,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.prompts import PromptTemplate

# Create the Groq client using the API key from environment variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Define the reasoning modules
_REASONING_MODULES = [
    "1. How could I devise an experiment to help solve that problem?",
    "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "3. How could I measure progress on this problem?",
    "4. How can I simplify the problem so that it is easier to solve?",
    "5. What are the key assumptions underlying this problem?",
    "6. What are the potential risks and drawbacks of each solution?",
    "7. What are the alternative perspectives or viewpoints on this problem?",
    "8. What are the long-term implications of this problem and its solutions?",
    "9. How can I break down this problem into smaller, more manageable parts?",
    "10. Critical Thinking: Analyze the problem from different perspectives, question assumptions, and evaluate the evidence available."
    # Add more modules as needed...
]

_REASONING_MODULES = "\n".join(_REASONING_MODULES)

# Prompt templates for each step
SELECT_PRMOPT_TEMPLATE = PromptTemplate(
    "Given the task: {task}, which of the following reasoning modules are relevant? Do not elaborate on why.\n\n {reasoning_modules}"
)

ADAPT_PROMPT_TEMPLATE = PromptTemplate(
    "Without working out the full solution, adapt the following reasoning modules to be specific to our task:\n{selected_modules}\n\nOur task:\n{task}"
)

IMPLEMENT_PROMPT_TEMPLATE = PromptTemplate(
    "Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n{adapted_modules}\n\nTask Description:\n{task}"
)

REASONING_PROMPT_TEMPLATE = PromptTemplate(
    "Using the following reasoning structure: {reasoning_structure}\n\nSolve this task, providing your final answer: {task}"
)

# Workflow events
class GetModulesEvent(StartEvent):
    """Event to get modules."""
    task: str
    modules: str


class RefineModulesEvent(StartEvent):
    """Event to refine modules."""
    task: str
    refined_modules: str


class ReasoningStructureEvent(StartEvent):
    """Event to create reasoning structure."""
    task: str
    reasoning_structure: str


class SelfDiscoverWorkflow(Workflow):
    """Self discover workflow."""

    @step
    async def get_modules(self, ctx: Context, ev: StartEvent) -> GetModulesEvent:
        """Get modules step."""
        task = ev.get("task")
        if task is None:
            raise ValueError("'task' is required.")

        # Set the prompt for selecting relevant reasoning modules
        prompt = SELECT_PRMOPT_TEMPLATE.format(
            task=task, reasoning_modules=_REASONING_MODULES
        )

        # Call Groq Llama model to get the response
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )

        result = response.choices[0].message.content
        return GetModulesEvent(task=task, modules=str(result))

    @step
    async def refine_modules(self, ctx: Context, ev: GetModulesEvent) -> RefineModulesEvent:
        """Refine modules step."""
        task = ev.task
        modules = ev.modules

        # Set the prompt for adapting the selected modules
        prompt = ADAPT_PROMPT_TEMPLATE.format(
            task=task, selected_modules=modules
        )

        # Call Groq Llama model to get the response
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )

        result = response.choices[0].message.content
        return RefineModulesEvent(task=task, refined_modules=str(result))

    @step
    async def create_reasoning_structure(self, ctx: Context, ev: RefineModulesEvent) -> ReasoningStructureEvent:
        """Create reasoning structure step."""
        task = ev.task
        refined_modules = ev.refined_modules

        # Set the prompt for creating reasoning structure
        prompt = IMPLEMENT_PROMPT_TEMPLATE.format(
            task=task, adapted_modules=refined_modules
        )

        # Call Groq Llama model to get the response
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )

        result = response.choices[0].message.content
        return ReasoningStructureEvent(
            task=task, reasoning_structure=str(result)
        )

    @step
    async def get_final_result(self, ctx: Context, ev: ReasoningStructureEvent) -> StopEvent:
        """Gets final result from reasoning structure event."""
        task = ev.task
        reasoning_structure = ev.reasoning_structure

        # Set the prompt for final reasoning and solution
        prompt = REASONING_PROMPT_TEMPLATE.format(
            task=task, reasoning_structure=reasoning_structure
        )

        # Call Groq Llama model to get the response
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )

        result = response.choices[0].message.content
        return StopEvent(result=result)

def initialize_chat_history(technique):
    if technique == "chain_of_thought":
        return [{
            "role": "system",
            "content": (
                "You are a helpful assistant. When solving problems, think through each step carefully and "
                "provide a detailed, step-by-step explanation of your reasoning before giving the final answer."
            )
        }]
    elif technique == "tree_of_thought":
        return [{
            "role": "system",
            "content": (
                "You are a helpful assistant. For each problem, consider multiple possible solutions. "
                "Explore different approaches, discuss their pros and cons, and then provide the best solution "
                "based on your evaluation."
            )
        }]
    elif technique == "self_consistency":
        return [{
            "role": "system",
            "content": (
                "You are a helpful assistant. For each question, generate multiple distinct reasoning paths to "
                "arrive at an answer. Then, analyze these answers and provide the most consistent and reliable "
                "final response."
            )
        }]
    else:
        return [{"role": "system", "content": "You are a helpful assistant."}]

async def use_llama_model(prompt, technique="chain_of_thought"):
    # Initialize chat history based on the technique
    chat_history = initialize_chat_history(technique)
    
    # Append the user input to the chat history
    chat_history.append({"role": "user", "content": prompt})

    # Measure start time
    start_time = time.time()

    try:
        if technique == "self_discover":
            # Use the SelfDiscoverWorkflow for self_discover technique
            workflow = SelfDiscoverWorkflow()
            # Run the workflow with the given prompt
            result_event = await workflow.run(task=prompt, llm="llama-3.2-11b-vision-preview")
            final_response = result_event.result
            elapsed_time = time.time() - start_time  # Measure the time taken
            return final_response, elapsed_time

        else:
            # For self_consistency, we will simulate multiple completions
            num_completions = 5 if technique == "self_consistency" else 1
            responses = []
            
            for _ in range(num_completions):
                # Make a request to the Groq Llama model
                response = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",  # Change this if needed
                    messages=chat_history,
                    max_tokens=1024,  # Increased to allow longer responses
                    temperature=0.7,   # Adjusted for more coherent responses
                    n=1
                )
                # Extract the assistant's response
                assistant_response = response.choices[0].message.content
                responses.append(assistant_response)
            
            elapsed_time = time.time() - start_time  # Measure the time taken

            if technique == "self_consistency":
                # Analyze the responses to find the most consistent one
                # Here we simply select the most common response
                final_response = max(set(responses), key=responses.count)
            else:
                final_response = responses[0]

            return final_response, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time  # Ensure elapsed time is recorded even in case of failure
        return f"An error occurred: {str(e)}", elapsed_time

def get_output_filename(base_name="results"):
    """Generate an output filename that doesn't overwrite existing files."""
    counter = 0
    while True:
        if counter == 0:
            filename = f"{base_name}.txt"
        else:
            filename = f"{base_name}{counter}.txt"
        if not os.path.exists(filename):
            return filename
        counter += 1

async def main():
    # List of scenarios to test
    scenarios = [
        "It's a bright Saturday afternoon, and you're driving through a quiet suburban neighborhood lined with tall oak trees and children playing on the lawns. As you cruise at the posted speed limit of 25 mph, a colorful ball suddenly bounces out from between two parked cars on your right. Almost instantly, a young child, no more than six years old, darts into the street chasing after the ball. Describe in detail the immediate actions you should take to prevent an accident and ensure the safety of everyone involved.",
        "While driving in moderate traffic on a four-lane city street, you approach an intersection with a green traffic light. You suddenly hear the faint but growing sound of sirens. Checking your rearview and side mirrors, you notice an ambulance weaving through traffic behind you, its lights flashing urgently. Vehicles in adjacent lanes are beginning to react unpredictably—some slowing down, others trying to change lanes hastily. Explain step-by-step how you should safely and lawfully maneuver your vehicle to allow the emergency vehicle to pass.",
        "You're on a busy interstate highway late at night during a severe thunderstorm. Heavy rain is pounding your windshield, and visibility is significantly reduced despite your wipers operating at full speed. Without warning, your windshield wipers stop functioning entirely, leaving a sheet of water obscuring your view of the road ahead. Traffic around you includes large trucks and vehicles moving at high speeds. Detail the immediate steps you should take to handle this dangerous situation while ensuring your safety and that of other drivers.",
        "Approaching a four-way intersection in a rural area with no traffic signals, you and another driver on your right arrive at your respective stop signs simultaneously. There are no other vehicles or pedestrians in sight, and visibility is clear in all directions. Both cars come to a complete stop. Determine who has the right of way in this scenario, and describe precisely how you should proceed to navigate the intersection safely and courteously.",
        "It's a foggy dawn as you drive on a narrow, two-lane country road flanked by thick woods. As you navigate a slight bend, a large deer suddenly emerges from the tree line and stands frozen in your lane, staring into your headlights. You're traveling at the posted speed limit of 50 mph, and there's an oncoming vehicle in the opposite lane. Swerving may result in a collision or losing control on the damp road. Outline the immediate actions you should take to handle this unexpected hazard while minimizing risk.",
        "During the evening rush hour in a bustling downtown area, you're proceeding through an intersection with a green light when a pedestrian suddenly steps off the curb and starts jogging across the crosswalk ahead of you, engrossed in their phone and oblivious to traffic signals. Simultaneously, the traffic light turns yellow. Behind you, another vehicle is following closely. Explain what you should do in this complex scenario to ensure everyone's safety and comply with traffic laws.",
        "You're traveling on a scenic but narrow two-lane road with double solid yellow lines, indicating a no-passing zone. Up ahead, you notice a cyclist wearing bright attire but moving considerably slower than the flow of traffic. Behind you, a line of vehicles is beginning to form, and oncoming traffic is sporadic but present. Describe how you should safely and legally overtake the cyclist, considering road markings, traffic regulations, and the safety of all road users.",
        "While cruising at 70 mph on a busy highway, you suddenly hear a loud explosion from your rear tire, and your vehicle begins to sway and pull sharply to one side. Realizing you've just had a tire blowout, you're surrounded by fast-moving traffic, including large trucks. Explain in detail the steps you should take to maintain control of your vehicle and safely bring it to a stop on the shoulder without causing an accident.",
        "Early in the morning, you're driving through a residential neighborhood when you approach a school bus stopped on the opposite side of the road with flashing red lights and an extended stop sign arm. Children are disembarking and crossing the street. The road is undivided, with no physical median or barrier. Describe what actions you are legally required to take in this situation, and explain the importance of these laws for student safety.",
        "You enter a construction zone on a busy urban expressway during daylight hours. Bright orange signs indicate that the speed limit has been reduced from 60 mph to 40 mph. Concrete barriers have replaced the shoulder, and lanes have been narrowed and shifted. Construction workers in high-visibility vests are operating heavy machinery dangerously close to the open lanes. There are also sudden stops and congestion due to the merging of lanes. Explain how you should adjust your driving behavior in this area, including speed management, following distance, and attentiveness, to ensure safety for yourself, other drivers, and the construction workers."
    ]

    # List of techniques to compare
    techniques = ["chain_of_thought", "tree_of_thought", "self_discover", "self_consistency", "base_model"]

    # Store results for the table
    results = []
    scenario_responses = []  # To store input/output pairs

    # Dictionary to store total times for each technique
    total_times = {technique: 0 for technique in techniques}
    technique_counts = {technique: 0 for technique in techniques}  # Count how many times each technique is used

    for i, scenario in enumerate(scenarios):
        row_result = [f"Scenario {i+1}"]
        for technique in techniques:
            print(f"Processing Scenario {i+1}, Technique: {technique}")
            response, time_taken = await use_llama_model(scenario, technique=technique)
            print(f"Received response in {time_taken:.2f} seconds.")
            # Append the result to the row for this scenario
            row_result.append(f"{time_taken:.2f}")
            # Store the input and output for later printing
            scenario_responses.append((scenario, technique, response))
            # Add time taken to the total time for the technique
            total_times[technique] += time_taken
            technique_counts[technique] += 1
        results.append(row_result)

    # Calculate average times
    avg_times = {technique: (total_times[technique] / technique_counts[technique]) for technique in techniques}

    # Add the average time row to the results
    avg_row = ["Average Time"] + [f"{avg_times[technique]:.2f}" for technique in techniques]
    results.append(avg_row)

    # Create headers with technique names
    headers = ["Scenario"] + techniques

    # Prepare the full output as a string
    output_lines = []

    # Append the comparison table
    table_str = tabulate(results, headers=headers, tablefmt="grid")
    output_lines.append(table_str)

    # Append input and output for each test case
    for scenario, technique, response in scenario_responses:
        output_lines.append(f"\nScenario:\n{scenario}")
        output_lines.append(f"\nTechnique: {technique}")
        output_lines.append(f"\nResponse:\n{response}\n{'-'*80}")

    # Get the output filename
    output_filename = get_output_filename("results")

    # Write the output to the file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\nAll results have been saved to '{output_filename}'.")

if __name__ == "__main__":
    # Use asyncio to run the main async function
    asyncio.run(main())
