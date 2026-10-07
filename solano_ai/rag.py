from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class RAGSystem:
    def __init__(self):
        print("Cargando sistema RAG...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu"
        )

        self.chunks = []
        self.embeddings = None

        self.load_documents()

    def load_documents(self):
        self.chunks = []

        KNOWLEDGE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        for file_path in KNOWLEDGE_DIR.glob("*.txt"):
            content = file_path.read_text(
                encoding="utf-8"
            )

            document_chunks = self.split_text(content)

            document_name = (
                file_path.stem
                .replace("_", " ")
                .strip()
            )

            for chunk in document_chunks:
                self.chunks.append(
                    {
                        "text": chunk,
                        "source": file_path.name,
                        "document_name": document_name
                    }
                )

        if not self.chunks:
            print(
                "RAG: no se encontraron documentos "
                "dentro de knowledge/."
            )
            return

        passages = []

        for chunk in self.chunks:
            passage = (
                f"passage: Documento {chunk['document_name']}. "
                f"{chunk['text']}"
            )

            passages.append(passage)

        self.embeddings = self.embedding_model.encode(
            passages,
            normalize_embeddings=True
        )

        print(
            f"RAG: {len(self.chunks)} fragmentos indexados."
        )

    def split_text(self, text, chunk_size=300):
        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n")
            if paragraph.strip()
        ]

        chunks = []

        for paragraph in paragraphs:

            if len(paragraph) <= chunk_size:
                chunks.append(paragraph)
                continue

            sentences = paragraph.split(". ")

            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()

                if not sentence:
                    continue

                candidate = (
                    current_chunk
                    + sentence
                    + ". "
                )

                if len(candidate) <= chunk_size:
                    current_chunk = candidate

                else:
                    if current_chunk:
                        chunks.append(
                            current_chunk.strip()
                        )

                    current_chunk = sentence + ". "

            if current_chunk:
                chunks.append(
                    current_chunk.strip()
                )

        return chunks

    def search(self, query, top_k=3):
        if not self.chunks or self.embeddings is None:
            return []

        formatted_query = f"query: {query}"

        query_embedding = self.embedding_model.encode(
            [formatted_query],
            normalize_embeddings=True
        )[0]

        scores = self.embeddings @ query_embedding

        ranked_indices = (
            scores.argsort()[::-1][:top_k]
        )

        results = []

        for index in ranked_indices:
            results.append(
                {
                    "text": self.chunks[index]["text"],
                    "source": self.chunks[index]["source"],
                    "score": float(scores[index])
                }
            )

        return results