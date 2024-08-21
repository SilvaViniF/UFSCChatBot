"""Using only faithfulness"""
#Passos:
#Carregar dataset:
#Rag no dataset -> salva resultados no output file
#Evaluation Prompt
#Define funcao de judge -> vai avaliar os resultados no output file


#Preciso adaptar isso pro llama que eu ja tenho.
import pandas as pd
import llama3,evaluate
import json,os
from tqdm.auto import tqdm
from typing import Optional, List, Tuple
import datasets


def run_rag_tests(
    eval_dataset: datasets.Dataset,
    output_file: str,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm(eval_dataset):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
          continue
        
        
       
        #answer, relevant_docs = answer_with_rag(question, llm, knowledge_index, reranker=reranker)
        """Substituir com meu método de gerar resposta:"""
        llm_answer=llama3nvidia.talk(question)
        answer = list(llm_answer)
        
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        result = {
            "question": question,
            "true_answer": example["answer"],
            "generated_answer": answer,
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)

evaluator_name = "Llama3-8B"
READER_MODEL_NAME=evaluator_name

def evaluate_answers(
    answer_path: str,
    evaluator_name: str,
) -> None:
    """Evaluates generated answers using the local Llama model."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r"))

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        # Prepare the input for the Llama model
        instruction = experiment["question"]
        response = experiment["generated_answer"]
        reference_answer = experiment["true_answer"]

        # Create a prompt for evaluation
        eval_prompt = f"""
        ###Task Description:
        Evaluate the following response based on the question and reference answer.
        
        ###The instruction to evaluate:
        {instruction}

        ###Response to evaluate:
        {response}

        ###Reference Answer (Score 5):
        {reference_answer}

        ###Score Rubrics:
        [Is the response correct, accurate, and factual based on the reference answer?]
        Score 1: The response is completely incorrect, inaccurate, and/or not factual.
        Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
        Score 3: The response is somewhat correct, accurate, and/or factual.
        Score 4: The response is mostly correct, accurate, and factual.
        Score 5: The response is completely correct, accurate, and factual.

        ###Feedback:
        """

        # Call the talk function to get evaluation feedback
        eval_result = list(llama3.talk(eval_prompt))  # Call the talk function directly
        feedback, score = eval_result[-1].strip().split("[RESULT]")  # Assuming the last output contains the result

        experiment[f"eval_score_{evaluator_name}"] = score.strip()
        experiment[f"eval_feedback_{evaluator_name}"] = feedback.strip()

    with open(answer_path, "w") as f:
        json.dump(answers, f)


if not os.path.exists("./output"):
    os.mkdir("./output")

for chunk_size in [200]:  # Add other chunk sizes (in tokens) as needed
    for embeddings in ["thenlper/gte-small"]:  # Add other embeddings as needed
        for rerank in [True, False]:
            settings_name = f"chunk:{chunk_size}_embeddings:{embeddings.replace('/', '~')}_rerank:{rerank}_reader-model:{READER_MODEL_NAME}"
            output_file_name = f"./output/rag_{settings_name}.json"

            print(f"Running evaluation for {settings_name}:")

            print("Loading knowledge base embeddings...")
            """knowledge_index = load_embeddings(
                RAW_KNOWLEDGE_BASE,
                chunk_size=chunk_size,
                embedding_model_name=embeddings,
            )"""
            

            print("Running RAG...")
            #reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0") if rerank else None
            run_rag_tests(
                eval_dataset=evaluate.eval_dataset,
                output_file=output_file_name,
                verbose=False,
                test_settings=settings_name,
            )

            print("Running evaluation...")
            evaluate_answers(
                output_file_name,
                evaluator_name,
            )


import glob

outputs = []
for file in glob.glob("./output/*.json"):
    output = pd.DataFrame(json.load(open(file, "r")))
    output["settings"] = file
    outputs.append(output)
result = pd.concat(outputs)