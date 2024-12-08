"""Using only faithfulness"""
#Passos:
#Carregar dataset:
#Rag no dataset -> salva resultados no output file
#Evaluation Prompt
#Define funcao de judge -> vai avaliar os resultados no output file
from flask import Flask
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)

import llama2,llama3
import pandas as pd
import json,os
from tqdm.auto import tqdm
from typing import Optional, List, Tuple
import datasets
import time
import random

def run_rag_tests(
    eval_dataset,
    output_file: str,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,
):
    try:
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    # Convert the dataset to a list and shuffle it
    eval_list = list(eval_dataset)
    # Select the first 10 questions
    selected_questions = eval_list

    for example in tqdm(selected_questions):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue
        
        print(f"Processing question: {question[:50]}...")

        # Call the talk function to get evaluation feedback
        #llm_answer_generator = llama3.talk(question, max_new_tokens=256)
        llm_answer_generator = llama2.talk(question, max_new_tokens=512)
        # Collect all the generated text, but we'll only keep the last chunk
        llm_answer = ""
        for chunk in llm_answer_generator:
            llm_answer = chunk  # Overwrite with each new chunk

        print(f"Final answer: {llm_answer[:100]}...")  # Print first 100 chars of final answer

        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {llm_answer}")
            print(f'True answer: {example["answer"]}')
        
        result = {
            "question": question,
            "true_answer": example["answer"],
            "generated_answer": llm_answer,
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)
        
        print(f"Saved result for question: {question[:50]}...")

    print("Completed run_rag_tests function")

evaluator_name = "llama2"
READER_MODEL_NAME=evaluator_name

def evaluate_answers(
    answer_path: str,
    evaluator_name: str,
) -> None:
    print(f"Starting evaluation process with {evaluator_name}")
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        print(f"Loading previous generations from {answer_path}")
        answers = json.load(open(answer_path, "r"))

    print(f"Total answers to evaluate: {len(answers)}")

    max_input_length = 4096  # Define a maximum input length, adjust as needed

    for i, experiment in enumerate(tqdm(answers)):
        print(f"\nProcessing experiment {i+1}/{len(answers)}")
        
        if f"eval_score_{evaluator_name}" in experiment:
            print(f"Experiment {i+1} already evaluated, skipping")
            continue

        instruction = experiment["question"]
        response = experiment["generated_answer"]
        reference_answer = experiment["true_answer"]
        
        print(f"Question: {instruction[:50]}...")  # Print first 50 chars of question
        print(f"Generated Answer: {response[:50]}...")  # Print first 50 chars of generated answer
        
        eval_prompt=f"""### Descrição da Tarefa:
        Avalie a seguinte resposta com base na pergunta e na resposta de referência.

        ### Instrução a ser avaliada:
        {instruction[:200]}

        ### Resposta a ser avaliada:
        {response[:200]}

        ### Resposta de Referência (Nota 5):
        {reference_answer[:200]}

        ### Critérios de Avaliação:
        [A resposta está correta, precisa e factual com base na resposta de referência?]
        Nota 1: A resposta está completamente incorreta, imprecisa e/ou não factual.
        Nota 2: A resposta está majoritariamente incorreta, imprecisa e/ou não factual.
        Nota 3: A resposta está parcialmente correta, precisa e/ou factual.
        Nota 4: A resposta está majoritariamente correta, precisa e factual.
        Nota 5: A resposta está completamente correta, precisa e factual.

        ### Avaliação:"""


        eval_prompt = eval_prompt[:max_input_length]
        print(f"Evaluation prompt length: {len(eval_prompt)}")
        print(f"eval_prompt:::::: {eval_prompt}")
        print("Calling llama3.eval() for evaluation...")
        start_time = time.time()
        eval_result_generator = llama3.eval(eval_prompt)
        eval_result = ""
        for chunk in eval_result_generator:
            eval_result = chunk  # Overwrite with each new chunk
        end_time = time.time()
        print(f"llama3.eval() completed in {end_time - start_time:.2f} seconds")
        
        eval_result_text = eval_result
        print(f"Evaluation result: {eval_result_text[:100]}...")  # Print first 100 chars of result
        import re
        try:
            feedback = eval_result_text
            if "Nota" in eval_result_text:
                # Usando regex para capturar o dígito após "Nota"
                match = re.search(r'Nota\s*([\d])', eval_result_text)
                if match:
                    score = match.group(1)  # Pega o primeiro dígito encontrado
                else:
                    score = "0"  # Score padrão se não encontrar
            else:
                feedback = eval_result_text
                score = "0"  # Score padrão
        except Exception as e:
            print(f"An error occurred: {e}")
            feedback = eval_result_text
            score = "0"  # Score padrão em caso de erro

    
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        print(f"Saving results for experiment {i+1}")
        with open(answer_path, "w") as f:
            json.dump(answers, f)

    print("Evaluation process completed")

if not os.path.exists("./output"):
    os.mkdir("./output")

for chunk_size in [200]:  # Add other chunk sizes (in tokens) as needed
    for embeddings in ["sentence-transformers/all-MiniLM-L6-v2"]:  # Add other embeddings as needed
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
                eval_dataset=datasets.load_dataset("SilvaFV/UFSC_eval_translated", split="train"),
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



if __name__ == "__main__":
   app.run(debug=True)


print("Finished, running visualize_judge...")
import visualize_judge
visualize_judge.main()