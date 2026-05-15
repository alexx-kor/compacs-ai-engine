import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Local vector store (JSON files under this directory)
    local_vector_store_dir: str = field(default="")

    # Ollama
    embed_model: str = os.getenv('EMBED_MODEL', 'nomic-embed-text')
    llm_model: str = os.getenv('LLM_MODEL', 'llama3.2:3b')
    
    # RAG
    chunk_size: int = int(os.getenv('CHUNK_SIZE', '1000'))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', '150'))
    top_k: int = int(os.getenv('TOP_K', '8'))
    rerank_top_k: int = int(os.getenv('RERANK_TOP_K', '3'))
    similarity_threshold: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.35'))
    batch_size: int = int(os.getenv('BATCH_SIZE', '32'))
    
    # Generation
    num_ctx: int = int(os.getenv('NUM_CTX', '4096'))
    num_predict: int = int(os.getenv('NUM_PREDICT', '400'))
    temperature: float = float(os.getenv('TEMPERATURE', '0.1'))
    top_p: float = float(os.getenv('TOP_P', '0.9'))
    repeat_penalty: float = float(os.getenv('REPEAT_PENALTY', '1.1'))
    
    # Limits
    max_text_length: int = int(os.getenv('MAX_TEXT_LENGTH', '3072'))
    min_chunk_size: int = int(os.getenv('MIN_CHUNK_SIZE', '100'))
    max_chunks_per_doc: int = int(os.getenv('MAX_CHUNKS_PER_DOC', '2000'))
    
    # Cache
    cache_enabled: bool = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
    cache_ttl: int = int(os.getenv('CACHE_TTL', '3600'))
    
    # Paths
    _project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    docs_folder: str = field(default='')
    few_shot_folder: str = field(default='')
    results_folder: str = field(default='')
    
    doc_files: list[tuple[str, str]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        if not self.local_vector_store_dir:
            self.local_vector_store_dir = str(self._project_root / "data" / "vector_store")
        if not self.docs_folder:
            self.docs_folder = str(self._project_root / 'doc-2.0-sources')
        if not self.few_shot_folder:
            self.few_shot_folder = str(self._project_root / 'data' / 'few_shot_examples')
        if not self.results_folder:
            self.results_folder = str(self._project_root / 'data' / 'results')

        if os.path.exists(self.docs_folder):
            for root, dirs, files in os.walk(self.docs_folder):
                for file in files:
                    if file.endswith(('.txt', '.md')):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(root, self.docs_folder)
                        if rel_path == '.':
                            source_name = file.replace('.txt', '').replace('.md', '')
                        else:
                            source_name = f"{rel_path}/{file}"
                        self.doc_files.append((full_path, source_name))
        
        os.makedirs(self.few_shot_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.local_vector_store_dir, exist_ok=True)


config = Config()