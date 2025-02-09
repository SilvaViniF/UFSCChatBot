import os,sys
import csv
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
from app import SearchService

search_service = SearchService()

def avalia(pergunta, resposta, resposta_ideal):
    prompt = f"""Analise cuidadosamente a pergunta, a resposta ideal e a resposta gerada fornecidas abaixo:

- **Pergunta:** {pergunta}  
- **Resposta ideal (referência):** {resposta_ideal}  
- **Resposta gerada:** {resposta}  

Avalie a qualidade da resposta gerada em uma escala de 0 a 1 com base nos seguintes critérios:

1. **Correção:** A resposta aborda completamente a pergunta e está alinhada à resposta ideal?  
2. **Precisão:** A resposta apresenta informações factualmente corretas e evita imprecisões?  
3. **Completude:** A resposta inclui todos os pontos essenciais fornecidos na resposta ideal, sem omissões importantes?  
4. **Clareza:** A resposta é clara, bem estruturada e compreensível?  

**Notas para avaliação:**
- 1.0: A resposta está completamente correta, precisa e alinhada à referência.  
- 0.75: A resposta está majoritariamente correta, mas contém imprecisões ou omissões menores.  
- 0.5: A resposta está parcialmente correta, mas apresenta erros significativos ou informações incompletas.  
- 0.25: A resposta está majoritariamente incorreta, com poucas partes corretas.  
- 0.0: A resposta está completamente incorreta ou não responde à pergunta.  

**Instrução final:**  
Responda apenas com o número decimal representando a pontuação final (exemplo: 0.75). Não escreva nada além do número decimal.  
Lembre que a resposta que você está avaliando deve estar na mesma língua de pergunta realizada.
"""
    
    messages = [{"role": "system", "content": prompt}]
    score_gen = search_service.generate(messages)   
    score = ""
    for chunk in score_gen:
        score += chunk
    return score

def process_csv_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            process_single_csv(file_path)

def process_single_csv(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        data = list(reader)

    df_questoes = pd.DataFrame(data[1:], columns=data[0])
    df = pd.read_csv(file_path, delimiter=',', quotechar='"', on_bad_lines='skip', encoding='utf-8')
    
    df['Score'] = ''
    
    for index, row in df.iterrows():
        resposta = row['Resposta']
        pergunta = row['Pergunta']
        
        resposta_ideal = df_questoes.loc[df_questoes['Pergunta'] == pergunta, 'Resposta_ideal'].iloc[0]

        score = avalia(pergunta, resposta, resposta_ideal)

        df.at[index, 'Score'] = score
    
    output_path = file_path.replace('.csv', '_scored.csv')
    df.to_csv(output_path, index=False)
    print(f"Processed and saved: {output_path}")

folder_path = "analysis/human_test/v2/llama3.1"

process_csv_files(folder_path)