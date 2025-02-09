import csv
import sys
import os
from txtai import Embeddings

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
from app import SearchService

embeddings = Embeddings(content=True, path="mixedbread-ai/mxbai-embed-large-v1")
SearchService.set_embeddings(embeddings)
search_service = SearchService()
search_service.index_chunks()


input_file = os.path.join('analysis/human_test', 'questoes.csv')
output_file = os.path.join('analysis/human_test', 'v2/llama3.1/questoes_with_answers.csv')

def get_database_answers(input_file: str, output_file: str):
    questions = []
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            questions.append(row)

    for question in questions:
        print(f"Processing question {question['ID']}: {question['Pergunta']}")
        answer_gen = search_service.talk(question['Pergunta'], 5)
        answer = ""
        for chunk in answer_gen:
            answer += chunk
        question['Resposta'] = answer
        print(f"Answer: {answer}\n")

    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['ID', 'Pergunta', 'Resposta', 'Resposta_ideal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for question in questions:
            row = {field: question.get(field, '') for field in fieldnames}
            writer.writerow(row)
    print(f"Results have been saved to {output_file}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/home/grupoh/.env")
    print(f"TESTING MODEL = {os.getenv('MODEL_ID')}")
    get_database_answers(input_file,output_file)