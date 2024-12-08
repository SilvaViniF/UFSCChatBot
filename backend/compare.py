import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# Define the folder containing JSON files
folder_path = 'output'
output_folder = 'analysis_results_llama3_v2'
ranking_image = 'json_ranking.png'

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Dictionary to store average scores
average_scores = {}

# Process each JSON file separately
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        json_path = os.path.join(folder_path, filename)
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure 'eval_score_Llama3-8B' is numeric
        df['eval_score_Llama3-8B'] = pd.to_numeric(df['eval_score_Llama3-8B'], errors='coerce')
        
        # Calculate average evaluation score
        avg_score = df['eval_score_Llama3-8B'].mean()
        average_scores[filename] = avg_score
        print(f"{avg_score}  {filename}")
        # Create distribution plot of evaluation scores
        plt.figure(figsize=(10, 6))
        sns.countplot(x='eval_score_Llama3-8B', data=df, hue='eval_score_Llama3-8B', palette='viridis', legend=False)
        plt.title(f'Distribution of Evaluation Scores - {filename}')
        plt.xlabel('Evaluation Score')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_folder, f'evaluation_scores_distribution_{filename}.png'))
        plt.close()
        
        # Additional Visualization: Boxplot of Evaluation Scores
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='eval_score_Llama3-8B', data=df, hue='eval_score_Llama3-8B', palette='viridis', legend=False)
        plt.title(f'Boxplot of Evaluation Scores - {filename}')
        plt.xlabel('Evaluation Score')
        plt.savefig(os.path.join(output_folder, f'evaluation_scores_boxplot_{filename}.png'))
        plt.close()

# Rank JSON files based on average evaluation scores
ranked_files = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)

# Create a horizontal bar chart for ranking
plt.figure(figsize=(14, 10), dpi=350)  # Increase figure size and DPI for higher resolution
ax = sns.barplot(x=[score for _, score in ranked_files], y=[filename for filename, _ in ranked_files], hue=[filename for filename, _ in ranked_files], palette='viridis', legend=False)
plt.title('Ranking of JSON Files Based on Average Evaluation Scores')
plt.xlabel('Average Evaluation Score')
plt.ylabel('JSON File')

# Customize X-axis ticks to include intervals like 1, 1.1, 1.2, etc.
max_score = max(average_scores.values())
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

plt.xlim(0, max_score + 0.1)  # Adjust the x-axis limit to include more space for ticks
plt.tight_layout()

# Apply grid settings
ax.grid(True)

plt.savefig(os.path.join(output_folder, ranking_image))
plt.close()

print("Analysis complete. Results saved in 'analysis_results' folder.")
print(f"Ranking chart saved to '{os.path.join(output_folder, ranking_image)}'.")
