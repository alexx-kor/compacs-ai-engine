import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from config import config


def load_prompt(filename: str) -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / filename
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


class FewShotLoader:
    @staticmethod
    def load_examples() -> List[Dict]:
        examples = []
        folder_path = Path(config.few_shot_folder)
        
        if not folder_path.exists():
            return []
        
        for file_path in folder_path.rglob("*"):
            if file_path.suffix == '.csv':
                try:
                    df = pd.read_csv(file_path)
                    if 'question' in df.columns and 'answer' in df.columns:
                        for _, row in df.iterrows():
                            examples.append({
                                'question': str(row['question']),
                                'answer': str(row['answer']),
                                'source': str(row.get('source', 'example'))
                            })
                except:
                    pass
            elif file_path.suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = [l.strip() for l in f if l.strip()]
                        for i in range(0, len(lines), 2):
                            if i+1 < len(lines):
                                examples.append({
                                    'question': lines[i],
                                    'answer': lines[i+1],
                                    'source': 'txt'
                                })
                except:
                    pass
        
        return examples
    
    @staticmethod
    def format_examples(examples: List[Dict], max_examples: int = 3) -> str:
        if not examples:
            return ""
        
        examples = examples[:max_examples]
        formatted = "\n\n##  EXAMPLES OF GOOD ANSWERS:\n\n"
        
        for i, ex in enumerate(examples, 1):
            formatted += f"**Example {i}:**\n"
            formatted += f"Question: {ex['question']}\n"
            formatted += f"Answer: {ex['answer']}\n"
            if ex.get('source'):
                formatted += f"Source: {ex['source']}\n"
            formatted += "\n"
        
        return formatted


RAG_API_PROMPT = load_prompt("rag_api_en.txt")
RAG_API_PARAMETER_PROMPT = load_prompt("rag_api_parameter_en.txt")
RAG_API_PARAMETERS_LIST_PROMPT = load_prompt("rag_api_parameters_list_en.txt")

DEFAULT_PROMPT = """You are a technical documentation expert. Answer based ONLY on the provided context.

FORMAT:
ANSWER: [clear, specific answer]
SOURCE: [document name, page X]

If not found: "NOT FOUND"
"""

FEW_SHOT_EXAMPLES = FewShotLoader.load_examples()


def get_relevant_examples(question: str, max_examples: int = 3) -> List[Dict]:
    if not FEW_SHOT_EXAMPLES:
        return []
    
    q_lower = question.lower()
    q_words = set(re.findall(r'\b\w{4,}\b', q_lower))
    
    scored_examples = []
    for ex in FEW_SHOT_EXAMPLES:
        ex_lower = ex['question'].lower()
        ex_words = set(re.findall(r'\b\w{4,}\b', ex_lower))
        
        if q_words and ex_words:
            score = len(q_words & ex_words) / len(q_words)
        else:
            score = 0
        
        scored_examples.append((score, ex))
    
    scored_examples.sort(key=lambda x: x[0], reverse=True)
    return [ex for score, ex in scored_examples[:max_examples] if score > 0]


def enhance_prompt_with_examples(base_prompt: str, question: str) -> str:
    if not FEW_SHOT_EXAMPLES:
        return base_prompt
    
    relevant_examples = get_relevant_examples(question, max_examples=3)
    if not relevant_examples:
        return base_prompt
    
    examples_text = FewShotLoader.format_examples(relevant_examples)
    return base_prompt + examples_text


def select_prompt(question: str) -> Tuple[str, int, float]:
    q_lower = question.lower()
    
    if any(kw in q_lower for kw in ['list of parameters', 'all parameters', 'parameter list']):
        base_prompt = RAG_API_PARAMETERS_LIST_PROMPT if RAG_API_PARAMETERS_LIST_PROMPT else DEFAULT_PROMPT
        num_predict, temperature = 1200, 0.05
    elif any(kw in q_lower for kw in ['parameter', 'param', 'field', 'difference']):
        base_prompt = RAG_API_PARAMETER_PROMPT if RAG_API_PARAMETER_PROMPT else DEFAULT_PROMPT
        num_predict, temperature = 800, 0.05
    else:
        base_prompt = RAG_API_PROMPT if RAG_API_PROMPT else DEFAULT_PROMPT
        num_predict, temperature = 1000, 0.1
    
    enhanced_prompt = enhance_prompt_with_examples(base_prompt, question)
    return enhanced_prompt, num_predict, temperature


class SmartPromptRouter:
    @staticmethod
    def select(question: str) -> Tuple[str, int, float]:
        return select_prompt(question)
    
    @staticmethod
    def get_examples_count() -> int:
        return len(FEW_SHOT_EXAMPLES)
    
    @staticmethod
    def reload_examples():
        global FEW_SHOT_EXAMPLES
        FEW_SHOT_EXAMPLES = FewShotLoader.load_examples()
        return len(FEW_SHOT_EXAMPLES)