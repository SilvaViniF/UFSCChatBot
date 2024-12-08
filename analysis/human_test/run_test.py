import csv
import sys
import os

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import llama2

# Set the working directory to the parent directory (backend)
os.chdir(parent_dir)

# Read the CSV file
input_file = os.path.join('human_test', 'questoes.csv')
output_file = os.path.join('human_test', 'questoes_with_answers.csv')

questions = []
with open(input_file, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        questions.append(row)

# Process each question and get the answer
for question in questions:
    print(f"Processing question {question['ID']}: {question['Pergunta']}")
    answer_gen = llama2.talk(question['Pergunta'])
    answer = ""
    for chunk in answer_gen:
        answer = chunk 
    question['Resposta'] = answer
    print(f"Answer: {answer}\n")

# Write the results to a new CSV file
with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['ID', 'Pergunta', 'Resposta', 'Resposta_ideal']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
    
    writer.writeheader()
    for question in questions:
        # Only write the fields we've specified
        row = {field: question.get(field, '') for field in fieldnames}
        writer.writerow(row)

print(f"Results have been saved to {output_file}")