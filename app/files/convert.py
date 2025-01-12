import csv
import json

def convert_csv_to_txt(input_csv_path, output_txt_path):
    data = []

    # Lendo o arquivo CSV
    with open(input_csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            entry = {
                "Nome": row.get("Nome", "").strip(),
                "Email": row.get("Email", "").strip(),
                "Telefone": [phone.strip() for phone in row.get("Telefone", "").split(",")],
                "Localizacao": row.get("Localização", "").strip(),
                "Cargo": row.get("Cargo", "").strip()
            }
            data.append(entry)

    # Gerando o texto formatado
    formatted_text = ""
    for entry in data:
        formatted_text += f"Professor: {entry['Nome']}\n"
        formatted_text += f"Email: {entry['Email']}\n"
        formatted_text += f"Telefone: {', '.join(entry['Telefone'])}\n"
        formatted_text += f"Localização: {entry['Localizacao']}\n"
        formatted_text += f"Cargo: {entry['Cargo']}\n"
        formatted_text += "\n"  # Separador entre entradas

    # Salvando em arquivo TXT
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(formatted_text)

    print(f"Arquivo TXT gerado com sucesso: {output_txt_path}")

# Exemplo de uso
input_csv_path = "ContatoProfessor_2.csv"  # Substitua pelo caminho do seu arquivo CSV
output_txt_path = "ContatoProfessor_2.txt"  # Caminho para o arquivo de saída em TXT
convert_csv_to_txt(input_csv_path, output_txt_path)
