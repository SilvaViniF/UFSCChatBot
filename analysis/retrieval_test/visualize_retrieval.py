import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_folder="/Users/viniciussilva/repos/UFSCChatBot/analysis/retrieval_test/retrieval_eval_results"
# Configurações gerais de estilo
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

# 🔹 Carregar os dados do JSON
with open("/Users/viniciussilva/repos/UFSCChatBot/analysis/retrieval_test/retrieval_evaluation_results.json", 'r') as file:
    data = json.load(file)

# 🔹 Extrair os resultados detalhados
df = pd.DataFrame(data['detailed_results'])

# 🔹 Calcular métricas
df['precisão'] = df['relevant_docs_count'] / df['total_docs_retrieved']
df['revocação'] = df['relevant_docs_count'] / df['relevant_docs_count'].max()
df['f1_score'] = 2 * (df['precisão'] * df['revocação']) / (df['precisão'] + df['revocação'])
df['f1_score'] = df['f1_score'].fillna(0)  # Evita divisão por zero

# 🔹 Média das métricas
avg_precisão = df['precisão'].mean()
avg_revocação = df['revocação'].mean()
avg_f1_score = df['f1_score'].mean()
avg_relevance_score = df['avg_relevance_score'].mean()

# 🔹 Criar DataFrame para gráficos
df_medias = pd.DataFrame({
    "Métrica": ["Precisão", "Revocação", "F1-Score"],
    "Valor": [avg_precisão, avg_revocação, avg_f1_score]
})

# 🔹 Criar gráfico de barras das métricas
plt.figure(figsize=(7, 5))
ax = sns.barplot(x="Métrica", y="Valor", data=df_medias, palette="Blues_r", edgecolor="black")
ax.set_ylabel("Média das Métricas")
ax.set_xlabel("")
ax.set_title("Métricas de Avaliação do Modelo")

# Adicionar os valores no topo das barras
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12)

# 🔹 Salvar gráfico
plt.tight_layout()
plt.savefig(output_folder+"/metrica_barras.png", dpi=300)
plt.savefig(output_folder+"/metrica_barras.pdf", dpi=300)
plt.show()

# 🔹 Criar gráfico de dispersão do score médio de relevância
plt.figure(figsize=(7, 5))
ax = sns.scatterplot(x=df.index, y=df["avg_relevance_score"], color="red", edgecolor="black")
ax.axhline(avg_relevance_score, color='blue', linestyle='--', label=f'Média = {avg_relevance_score:.2f}')
ax.set_xlabel("Consultas")
ax.set_ylabel("Score Médio de Relevância")
ax.set_title("Distribuição dos Scores Médios de Relevância")
ax.legend()

# 🔹 Salvar gráfico
plt.tight_layout()
plt.savefig(output_folder+"/relevancia_dispersao.png", dpi=300)
plt.savefig(output_folder+"/relevancia_dispersao.pdf", dpi=300)
plt.show()

print(f"Average Precision: {avg_precisão:.4f}")
print(f"Average Recall: {avg_revocação:.4f}")
print(f"Average F1-Score: {avg_f1_score:.4f}")
print(f"Average Relevance Score: {avg_relevance_score:.4f}")