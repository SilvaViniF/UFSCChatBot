import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def read_json_files(base_path):
    data = []
    json_files = []
    for folder in os.listdir(base_path):
        if folder.startswith('output'):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.json'):
                        file_path = os.path.join(folder_path, filename)
                        with open(file_path, 'r') as file:
                            json_data = json.load(file)
                            data.append((file_path, json_data))
                        json_files.append(file_path)
    return data, json_files

def process_data(data):
    eval_scores = defaultdict(list)
    test_settings = defaultdict(int)
    file_avg_scores = {}

    for file_path, file_data in data:
        file_scores = []
        for item in file_data:
            for key, value in item.items():
                if key.startswith('eval_score_'):
                    model = key.split('_')[-1]
                    score = int(value)
                    eval_scores[model].append(score)
                    file_scores.append(score)
            
            if 'test_settings' in item:
                test_settings[item['test_settings']] += 1
        
        if file_scores:
            file_avg_scores[file_path] = sum(file_scores) / len(file_scores)

    return eval_scores, test_settings, file_avg_scores

def create_visualizations(eval_scores, test_settings, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Evaluation scores distribution
    plt.figure(figsize=(12, 6))
    for model, scores in eval_scores.items():
        sns.kdeplot(scores, label=model)
    plt.title('Distribution of Evaluation Scores')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'eval_scores_distribution.png'))
    plt.close()

    # Average scores by model
    avg_scores = {model: sum(scores) / len(scores) for model, scores in eval_scores.items()}
    plt.figure(figsize=(12, 6))
    plt.bar(avg_scores.keys(), avg_scores.values())
    plt.title('Average Evaluation Scores by Model')
    plt.xlabel('Model')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'avg_scores_by_model.png'))
    plt.close()

def print_json_info(json_files, file_avg_scores):
    print(f"Number of JSON files analyzed: {len(json_files)}")
    print("\nList of JSON files and their average scores:")
    for file_path in json_files:
        avg_score = file_avg_scores.get(file_path, "N/A")
        if isinstance(avg_score, float):
            avg_score = f"{avg_score:.2f}"
        print(f"- {file_path}: Average Score = {avg_score}")

def main():
    base_path = '.'  # Current directory
    output_folder = 'judge_results'

    data, json_files = read_json_files(base_path)
    eval_scores, test_settings, file_avg_scores = process_data(data)
    create_visualizations(eval_scores, test_settings, output_folder)
    
    print_json_info(json_files, file_avg_scores)

if __name__ == '__main__':
    main()