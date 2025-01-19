import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Add this import at the top

# Load the JSON data
with open('retrieval_evaluation_results.json', 'r') as file:
    data = json.load(file)

# Extract the detailed_results
results = data['detailed_results']

# Create a DataFrame
df = pd.DataFrame(results)

# Create the directory if it doesn't exist
if not os.path.exists('retrieval_eval_results'):
    os.makedirs('retrieval_eval_results')

# Function to create and save a plot
def save_plot(fig, filename):
    plt.tight_layout()
    plt.savefig(f'retrieval_eval_results/{filename}')  # Updated to save in the specified folder
    plt.close(fig)

# 1. Retrieval Success Rate
fig, ax = plt.subplots(figsize=(10, 6))
success_rate = df['retrieval_success'].mean() * 100
ax.bar(['Retrieval Success Rate'], [success_rate])
ax.set_ylim(0, 100)
ax.set_ylabel('Percentage')
ax.set_title('Retrieval Success Rate')
for i, v in enumerate([success_rate]):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center')
save_plot(fig, 'retrieval_success_rate.png')

# 2. Distribution of Relevant Documents
fig, ax = plt.subplots(figsize=(10, 6))
df['relevant_docs_count'].value_counts().sort_index().plot(kind='bar', ax=ax)
ax.set_xlabel('Number of Relevant Documents')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Relevant Documents')
save_plot(fig, 'relevant_docs_distribution.png')

# 3. Average Relevance Score Distribution
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['avg_relevance_score'], kde=True, ax=ax)
ax.set_xlabel('Average Relevance Score')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Average Relevance Scores')
save_plot(fig, 'avg_relevance_score_distribution.png')

# 4. Relevance Scores Box Plot
fig, ax = plt.subplots(figsize=(10, 6))
df['relevance_scores'].apply(pd.Series).melt().dropna().plot(kind='box', ax=ax)
ax.set_ylabel('Relevance Score')
ax.set_title('Box Plot of Relevance Scores')
save_plot(fig, 'relevance_scores_boxplot.png')

# 5. Scatter plot: Relevant Docs vs Avg Relevance Score
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['relevant_docs_count'], df['avg_relevance_score'])
ax.set_xlabel('Number of Relevant Documents')
ax.set_ylabel('Average Relevance Score')
ax.set_title('Relevant Documents vs Average Relevance Score')
save_plot(fig, 'relevant_docs_vs_avg_score.png')

print("Visualizations have been generated and saved as PNG files.")