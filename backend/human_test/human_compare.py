import os,sys
import csv
import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from llama318B import eval  # Assuming you have this import available

def avalia(pergunta, resposta, resposta_ideal):
    prompt = f"""Analise cuidadosamente a pergunta, a resposta ideal e a resposta gerada fornecidas abaixo:

Pergunta: {pergunta}
Resposta ideal:(NOTA 1) {resposta_ideal}
Resposta gerada: {resposta}

Avalie a qualidade da resposta gerada em uma escala de 0 a 1, considerando os seguintes critérios:

A resposta está correta, precisa e factual com base na resposta de referência?
Nota 1: A resposta está completamente incorreta, imprecisa e/ou não factual.
Nota 2: A resposta está majoritariamente incorreta, imprecisa e/ou não factual.
Nota 3: A resposta está parcialmente correta, precisa e/ou factual.
Nota 4: A resposta está majoritariamente correta, precisa e factual.
Nota 5: A resposta está completamente correta, precisa e factual.

Pondere esses critérios e forneça uma pontuação final única entre 0 (qualidade muito baixa) e 1 (qualidade excelente).

Responda apenas com o número decimal representando a pontuação final. Não escreva nada alem do número decimal."""

    score_gen = eval(prompt)   
    score = ""
    for chunk in score_gen:
        score = chunk
    return score


def process_csv_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            process_single_csv(file_path)

def process_single_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Load questoes with resposta ideal
    with open('questoes.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        data = list(reader)

    df_questoes = pd.DataFrame(data[1:], columns=data[0])
    
    # Create a new 'Score' column
    df['Score'] = ''
    
    # Process each row
    for index, row in df.iterrows():
        resposta = row['Resposta']
        pergunta = row['Pergunta']
        
        # Fetch resposta_ideal from df_questoes
        resposta_ideal = df_questoes.loc[df_questoes['Pergunta'] == pergunta, 'Resposta_ideal'].iloc[0]

        score = avalia(pergunta, resposta, resposta_ideal)

        df.at[index, 'Score'] = score
    
    # Save the updated CSV
    output_path = file_path.replace('.csv', '_scored.csv')
    df.to_csv(output_path, index=False)
    print(f"Processed and saved: {output_path}")




# Specify the folder path containing your CSV files'
folder_path = '/home/grupoh/backend/human_test/llama3_chunk1024_100'

# Run the processing
process_csv_files(folder_path)