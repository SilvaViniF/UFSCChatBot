import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_statistics(file_path):
    """
    Reads a CSV file and calculates statistics on the 'Score' column.
    Returns a tuple with a dictionary of statistics and a Series of valid scores.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        if 'Score' not in df.columns:
            print(f"Skipping {file_path}: no 'Score' column found.")
            return None
        
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        df = df.dropna(subset=['Score'])
        
        if df.empty:
            print(f"No valid scores in {file_path}.")
            return None
        
        stats = {
            'mean': df['Score'].mean(),
            'median': df['Score'].median(),
            'std': df['Score'].std(),
            'min': df['Score'].min(),
            'max': df['Score'].max(),
            'count': df['Score'].count()
        }
        return stats, df['Score']
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_folders(main_folder):
    """
    Walks through the main folder and subfolders, processes CSV files,
    and collects statistics and scores.
    Returns a dictionary where each key is the file path and each value
    is a dictionary with the computed statistics and scores.
    """
    results = {}
    for root, _, files in os.walk(main_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                stats_and_scores = calculate_statistics(file_path)
                if stats_and_scores is not None:
                    stats, scores = stats_and_scores
                    results[file_path] = {'stats': stats, 'scores': scores}
    return results

def save_results(results, output_file='average_scores.txt'):
    """
    Saves a summary of the statistics to a text file.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path, data in results.items():
            stats = data['stats']
            outfile.write(
                f"{file_path}:\n"
                f"  Mean   = {stats['mean']:.2f}\n"
                f"  Median = {stats['median']:.2f}\n"
                f"  Std    = {stats['std']:.2f}\n"
                f"  Min    = {stats['min']:.2f}\n"
                f"  Max    = {stats['max']:.2f}\n"
                f"  Count  = {stats['count']}\n\n"
            )

def plot_average_scores(results, output_image='average_scores.png'):
    """
    Plots a bar chart of the average scores for each CSV file.
    """
    file_labels = []
    means = []
    for file_path, data in results.items():
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_labels.append(folder_name)
        means.append(data['stats']['mean'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=file_labels, y=means, palette='viridis')
    plt.xlabel("LLM")
    plt.ylabel("Score médio")
    plt.title("Score médio para cada LLM")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.show()

def plot_score_distributions(results, output_image='score_distributions.png'):
    """
    Creates a box plot showing the distribution of scores per CSV file.
    """
    data_list = []
    for file_path, data in results.items():
        folder_name = os.path.basename(os.path.dirname(file_path))
        for score in data['scores']:
            data_list.append({'file': folder_name, 'score': score})
    
    if not data_list:
        print("No data available for plotting score distributions.")
        return
    
    combined_df = pd.DataFrame(data_list)
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='file', y='score', data=combined_df, palette='Set2')
    plt.xlabel("LLM")
    plt.ylabel("Score")
    plt.title("Distribuição de Score para cada LLM")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.show()

def main():
    main_folder = 'analysis'
    results = process_folders(main_folder)
    
    if results:
        save_results(results)
        plot_average_scores(results)
        plot_score_distributions(results)
    else:
        print("No CSV files were processed.")

if __name__ == '__main__':
    main()