import os
import csv
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statistics import mean, stdev

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

def parse_model_info(file_path):
    """Extrai informações do modelo a partir do caminho do arquivo"""
    match = re.search(r'/(v\d+)/([^/]+)/questoes', file_path)
    version = match.group(1) if match else 'v1'
    model_name = match.group(2) if match else 'unknown'
    
    # Extrai parâmetros do modelo
    params = {}
    if 'chunk' in model_name:
        chunk_match = re.search(r'chunk(\d+)_(\d+)', model_name)
        if chunk_match:
            params['chunk_size'] = int(chunk_match.group(1))
            params['chunk_overlap'] = int(chunk_match.group(2))
            model = model_name.split('_chunk')[0]
    elif 'k=' in file_path:
        k_match = re.search(r'k=(\d+)', file_path)
        params['k'] = int(k_match.group(1)) if k_match else None
        model = model_name.split('_')[0]
    else:
        model = model_name
    
    return {
        'model': model,
        'version': version,
        **params
    }

def calculate_stats(file_path):
    """Calcula estatísticas descritivas"""
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'Score' not in reader.fieldnames:
                return None
            
            scores = []
            for row in reader:
                try:
                    score = float(row['Score'].split('\n')[0])
                    scores.append(score)
                except ValueError:
                    continue
            
            if not scores:
                return None
            
            return {
                'mean': mean(scores),
                'std': stdev(scores) if len(scores) > 1 else 0,
                'n': len(scores),
                'ci95': 1.96 * (stdev(scores)/len(scores)**0.5) if len(scores) > 1 else 0
            }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def generate_visualizations(df):
    """Gera visualizações acadêmicas multivariadas"""
    
    # Configurações comuns
    plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
    
    # 1. Gráfico de comparação principal com IC
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='model', y='mean', hue='version', data=df, 
                    ci='sd', errwidth=1.5, capsize=0.1)
    plt.title('Comparação de Desempenho entre Modelos LLM\nMédia com Intervalo de Confiança de 95%', 
             fontsize=14, pad=20)
    plt.xlabel('Modelos', fontsize=12, labelpad=15)
    plt.ylabel('Pontuação Média', fontsize=12, labelpad=15)
    plt.ylim(0, 0.5)
    ax.legend(title='Versão', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('fig1_comparacao_modelos_ic.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Gráfico de dispersão com tamanho amostral
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x='mean', y='std', size='n', hue='model',
                             data=df, sizes=(50, 300), alpha=0.8, palette='Dark2')
    plt.title('Relação entre Média, Variabilidade e Tamanho Amostral', fontsize=14)
    plt.xlabel('Média de Pontuação', fontsize=12)
    plt.ylabel('Desvio Padrão', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('fig2_dispersao_metricas.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Gráfico de parâmetros para Llama3
    if 'k' in df.columns:
        llama_df = df[df['model'] == 'llama3'].dropna(subset=['k'])
        if not llama_df.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='k', y='mean', data=llama_df, marker='o', 
                         ci='sd', err_style='band', color='#2ca02c')
            plt.fill_between(llama_df['k'], 
                            llama_df['mean'] - llama_df['ci95'], 
                            llama_df['mean'] + llama_df['ci95'], 
                            alpha=0.2, color='#2ca02c')
            plt.title('Desempenho do Llama3 em Função do Parâmetro k\nCom Intervalo de Confiança', 
                     fontsize=14)
            plt.xlabel('Valor de k (Número de Documentos Recuperados)', fontsize=12)
            plt.ylabel('Pontuação Média', fontsize=12)
            plt.xticks(llama_df['k'].unique())
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('fig3_llama3_k_analysis.png', dpi=300)
            plt.close()

    # 4. Heatmap de correlação de métricas
    numeric_df = df[['mean', 'std', 'n', 'ci95']].corr()
    print(numeric_df)
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df, annot=True, cmap='coolwarm', fmt=".2f",
               cbar_kws={'label': 'Coeficiente de Correlação'})
    plt.title('Matriz de Correlação entre Métricas de Desempenho', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('fig4_correlacao_metricas.png', dpi=300)
    plt.close()

    # 5. Distribuição comparativa em boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='model', y='mean', data=df, showmeans=True,
               meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'red'})
    plt.title('Distribuição das Médias por Modelo', fontsize=14)
    plt.xlabel('Modelos', fontsize=12)
    plt.ylabel('Pontuação Média', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('fig5_distribuicao_medias.png', dpi=300)
    plt.close()

def generate_latex_table(df):
    """Gera tabela LaTeX para inclusão no documento"""
    df = df.sort_values(by='mean', ascending=False)
    latex = df[['model', 'version', 'mean', 'std', 'n']].to_latex(
        index=False,
        column_format='lccrr',
        header=['Modelo', 'Versão', 'Média', 'Desvio Padrão', 'n'],
        float_format="%.3f",
        caption="Resultados comparativos dos modelos de LLM",
        label="tab:resultados_modelos"
    )
    with open('tabela_resultados.tex', 'w', encoding='utf-8') as f:
        f.write(latex)

def process_folders(main_folder):
    """Processa diretórios e gera análise"""
    data = []
    
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('_scored.csv'):
                file_path = os.path.join(root, file)
                stats = calculate_stats(file_path)
                if stats:
                    model_info = parse_model_info(file_path)
                    data.append({
                        **model_info,
                        **stats,
                        'file_path': file_path
                    })
    
    df = pd.DataFrame(data)
    
    if not df.empty:
        generate_visualizations(df)
        generate_latex_table(df)
        df.to_csv('analise_completa.csv', index=False)
        print("Análise concluída. Arquivos gerados:")
        print("- comparacao_modelos.png")
        print("- llama3_k_analysis.png (se aplicável)")
        print("- tabela_resultados.tex")
        print("- analise_completa.csv")

# Execução principal
if __name__ == "__main__":
    main_folder_path = 'analysis'
    process_folders(main_folder_path)