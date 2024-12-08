import os
import csv
from statistics import mean

def calculate_average_score(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'Score' not in reader.fieldnames:
                return None  # Ignore files without 'Score' column
            scores = []
            for row in reader:
                try:
                    score = float(row['Score'].split('\n')[0])  # Take only the first line
                    scores.append(score)
                except ValueError:
                    print(f"Warning: Invalid score in {file_path}: {row['Score']}")
        return mean(scores) if scores else None
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_folders(main_folder):
    results = []
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                avg_score = calculate_average_score(file_path)
                if avg_score is not None:
                    results.append(f"{file_path}: {avg_score:.2f}")
    
    with open('average_scores.txt', 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(results))

# Use the current directory as the main folder path
main_folder_path = '.'
process_folders(main_folder_path)