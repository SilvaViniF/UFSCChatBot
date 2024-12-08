import json
from typing import List, Tuple
from tqdm import tqdm
import llama3  # Assuming this is your module with the search and LLM functions

def judge_relevance_llm(question: str, true_answer: str, doc: str) -> float:
    """Use LLM to judge the relevance of a retrieved document."""
    prompt = f"""Pergunta: {question}
Resposta Correta: {true_answer}
Documento Recuperado: {doc}

Em uma escala de 0 a 1, quão relevante é o documento recuperado para responder à pergunta com precisão?
Considere o seguinte:
- O documento contém informações diretamente relacionadas à pergunta?
- O documento fornece contexto que poderia levar à resposta correta?
- As informações no documento são consistentes com a resposta correta?

Forneça sua avaliação como um número decimal entre 0 (completamente irrelevante) e 1 (altamente relevante).
Responda apenas com o número decimal."""
    
    response = llama3.eval(prompt)  # Assuming 'eval' is your function to call the LLM
    try:
        response=list(response)
        response=response[-1]
        score = float(response.strip())
        return max(0, min(score, 1))  # Ensure the score is between 0 and 1
    except ValueError:
        print(f"Invalid response from LLM: {response}")
        return 0

def evaluate_retrieval(dataset: List[dict], k: int = 5) -> List[dict]:
    """Evaluate the retrieval performance for each question in the dataset."""
    results = []
    for item in tqdm(dataset, desc="Evaluating Retrieval"):
        question = item['question']
        true_answer = item['answer']
        
        # Get retrieved documents
        retrieved_docs = llama3.search(question, llama3.ST, llama3.faiss_index, llama3.bm25, k=k)
        
        relevance_scores = []
        relevant_docs = []
        for _, _, doc in retrieved_docs:
            relevance_score = judge_relevance_llm(question, true_answer, doc)
            relevance_scores.append(relevance_score)
            if relevance_score > 0.5:  # You can adjust this threshold
                relevant_docs.append(doc)
        
        avg_relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        retrieval_success = len(relevant_docs) > 0

        results.append({
            'question': question,
            'true_answer': true_answer,
            'retrieved_docs': [doc for _, _, doc in retrieved_docs],
            'relevant_docs_count': len(relevant_docs),
            'total_docs_retrieved': len(retrieved_docs),
            'retrieval_success': retrieval_success,
            'avg_relevance_score': avg_relevance_score,
            'relevance_scores': relevance_scores,
            'max_relevance_score': max(relevance_scores) if relevance_scores else 0
        })
    
    return results

def calculate_metrics(eval_results: List[dict]) -> dict:
    """Calculate retrieval metrics based on LLM judgments."""
    total_questions = len(eval_results)
    successful_retrievals = sum(result['retrieval_success'] for result in eval_results)
    
    avg_relevance = sum(result['avg_relevance_score'] for result in eval_results) / total_questions
    max_relevance = sum(result['max_relevance_score'] for result in eval_results) / total_questions
    
    precision_at_k = sum(result['relevant_docs_count'] / result['total_docs_retrieved'] for result in eval_results) / total_questions
    recall_at_k = sum(result['retrieval_success'] for result in eval_results) / total_questions
    
    f1_score = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if precision_at_k + recall_at_k > 0 else 0
    
    return {
        'total_questions': total_questions,
        'successful_retrievals': successful_retrievals,
        'average_relevance': avg_relevance,
        'max_relevance': max_relevance,
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'f1_score': f1_score
    }

def main():
    # Load your evaluation dataset
    with open('UFSC_eval_translated.json', 'r', encoding='utf-8') as f:
        eval_dataset = json.load(f)
    
    # Evaluate retrieval
    eval_results = evaluate_retrieval(eval_dataset, k=10)  # Adjust k as needed
    
    # Calculate metrics
    metrics = calculate_metrics(eval_results)
    
    # Print results
    print("Retrieval Evaluation Results:")
    print(f"Total Questions Evaluated: {metrics['total_questions']}")
    print(f"Successful Retrievals: {metrics['successful_retrievals']}")
    print(f"Average Relevance: {metrics['average_relevance']:.2f}")
    print(f"Max Relevance: {metrics['max_relevance']:.2f}")
    print(f"Precision@k: {metrics['precision_at_k']:.2f}")
    print(f"Recall@k: {metrics['recall_at_k']:.2f}")
    print(f"F1 Score: {metrics['f1_score']:.2f}")
    
    # Save detailed results to a file
    with open('retrieval_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'detailed_results': eval_results,
            'summary_metrics': metrics
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()