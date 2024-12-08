from tqdm.auto import tqdm
import pandas as pd
import json
import datasets
from huggingface_hub import InferenceClient, login, create_repo
import config
import os
import pickle

# Hugging Face login
login(token=config.token)

pd.set_option("display.max_colwidth", None)

# Load dataset
ds = datasets.load_dataset("SilvaFV/UFSCdatabase", split="train")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

# Split documents
langchain_docs = [LangchainDocument(page_content=doc["content"], metadata={"Title": doc["Title"]}) for doc in tqdm(ds)]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

docs_processed = []
for doc in langchain_docs:
    docs_processed += text_splitter.split_documents([doc])
    

# Initialize Inference Client
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm_client = InferenceClient(model=repo_id, timeout=120)

def call_llm(inference_client: InferenceClient, prompt: str):
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]

QA_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::
"""

import random

N_GENERATIONS = 10
outputs_file = "outputs.pkl"

# Load existing outputs if file exists
if os.path.exists(outputs_file):
    with open(outputs_file, "rb") as f:
        outputs = pickle.load(f)
else:
    outputs = []

# Determine remaining generations
remaining_generations = N_GENERATIONS - len(outputs)
if remaining_generations > 0:
    for sampled_context in tqdm(random.sample(docs_processed, remaining_generations)):
        output_QA_couple = call_llm(llm_client, QA_generation_prompt.format(context=sampled_context.page_content))
        try:
            question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
            answer = output_QA_couple.split("Answer: ")[-1]
            assert len(answer) < 300, "Answer is too long"
            outputs.append(
                {
                    "context": sampled_context.page_content,
                    "question": question,
                    "answer": answer,
                    "source_doc": sampled_context.metadata["Title"],
                }
            )
            # Save outputs to file
            with open(outputs_file, "wb") as f:
                pickle.dump(outputs, f)
        except:
            continue

question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """


question_relevance_critique_prompt = """
You will receive a question.
Your task is to provide a 'total rating' representing how useful this question can be for answering queries from students in the automation course at UFSC Blumenau.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """


question_standalone_critique_prompt = """
You will receive a question.
Your task is to provide a 'total rating' representing how independent from context this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For example, if the question refers to a specific setting, like 'in the context' or 'in the document', the rating should be 1.
The questions can contain technical nouns or obscure acronyms like SIARE, UFSC and still receive a rating of 5: it just needs to be clear to an operator with access to documentation what the question is about.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

# File to save critiques
critiques_file = "critiques.pkl"

# Load existing critiques if file exists
if os.path.exists(critiques_file):
    with open(critiques_file, "rb") as f:
        critiques = pickle.load(f)
else:
    critiques = []


remaining_outputs = [output for output in outputs if output not in critiques]

print(f"Generating critique for {len(remaining_outputs)} QA couples...")

for output in tqdm(remaining_outputs):
    evaluations = {
        "groundedness": call_llm(
            llm_client,
            question_groundedness_critique_prompt.format(context=output["context"], question=output["question"]),
        ),
        "relevance": call_llm(
            llm_client,
            question_relevance_critique_prompt.format(question=output["question"]),
        ),
        "standalone": call_llm(
            llm_client,
            question_standalone_critique_prompt.format(question=output["question"]),
        ),
    }
    try:
        for criterion, evaluation in evaluations.items():
            score, eval = (
                int(evaluation.split("Total rating: ")[-1].strip()),
                evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
            )
            output.update(
                {
                    f"{criterion}_score": score,
                    f"{criterion}_eval": eval,
                }
            )
        
        # Save the critique for this output
        critiques.append(output)

        # Save critiques to file
        with open(critiques_file, "wb") as f:
            pickle.dump(critiques, f)
            
    except Exception as e:
        continue

import pandas as pd

pd.set_option("display.max_colwidth", None)

generated_questions = pd.DataFrame.from_dict(critiques)

eval_dataset = datasets.Dataset.from_pandas(generated_questions, split="train", preserve_index=False)

# Create new dataset repository
repo_id = "SilvaFV/UFSC_eval"

# Push dataset to Hugging Face Hub
eval_dataset.push_to_hub(repo_id)
