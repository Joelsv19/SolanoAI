from pathlib import Path

from pypdf import PdfReader
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
        self.embeddings = None

        KNOWLEDGE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # Cargar archivos TXT
        for file_path in KNOWLEDGE_DIR.glob("*.txt"):
            self.load_txt_file(file_path)

        # Cargar archivos PDF
        for file_path in KNOWLEDGE_DIR.glob("*.pdf"):
            self.load_pdf_file(file_path)

        if not self.chunks:
            print(
                "RAG: no se encontraron documentos "
                "dentro de knowledge/."
            )
            return

        passages = []

        for chunk in self.chunks:
            document_name = (
                Path(chunk["source"])
                .stem
                .replace("_", " ")
            )

            page_text = ""

            if chunk["page"] is not None:
                page_text = (
                    f" Página {chunk['page']}."
                )

            passage = (
                f"passage: Documento {document_name}."
                f"{page_text} "
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

    def load_txt_file(self, file_path):
        try:
            content = file_path.read_text(
                encoding="utf-8"
            )

            document_chunks = self.split_text(
                content
            )

            for chunk in document_chunks:
                self.chunks.append(
                    {
                        "text": chunk,
                        "source": file_path.name,
                        "page": None
                    }
                )

            print(
                f"TXT cargado: {file_path.name}"
            )

        except Exception as error:
            print(
                f"Error leyendo {file_path.name}: "
                f"{error}"
            )

    def load_pdf_file(self, file_path):
        try:
            reader = PdfReader(
                str(file_path)
            )

            pages_loaded = 0

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):
                text = page.extract_text()

                if not text:
                    continue

                text = text.strip()

                if not text:
                    continue

                document_chunks = self.split_text(
                    text
                )

                for chunk in document_chunks:
                    self.chunks.append(
                        {
                            "text": chunk,
                            "source": file_path.name,
                            "page": page_number
                        }
                    )

                pages_loaded += 1

            print(
                f"PDF cargado: {file_path.name} "
                f"({pages_loaded} páginas con texto)"
            )

        except Exception as error:
            print(
                f"Error leyendo PDF "
                f"{file_path.name}: {error}"
            )

    def split_text(
        self,
        text,
        chunk_size=900,
        overlap=180
    ):
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return []

        chunks = []
        current_chunk = ""

        for line in lines:
            candidate = (
                current_chunk + "\n" + line
                if current_chunk
                else line
            )

            if len(candidate) <= chunk_size:
                current_chunk = candidate

            else:
                if current_chunk:
                    chunks.append(
                        current_chunk.strip()
                    )

                previous_text = (
                    current_chunk[-overlap:]
                    if current_chunk
                    else ""
                )

                current_chunk = (
                    previous_text + "\n" + line
                    if previous_text
                    else line
                )

        if current_chunk:
            chunks.append(
                current_chunk.strip()
            )

        return chunks

    def reload_documents(self):
         print("\nRecargando base de conocimiento...")

         self.load_documents()
 
         return self.get_documents_info()


    def get_documents_info(self):
        documents = {}

        for chunk in self.chunks:
            source = chunk["source"]
            page = chunk.get("page")

            if source not in documents:
                documents[source] = {
                    "source": source,
                    "chunks": 0,
                    "pages": set()
                }

            documents[source]["chunks"] += 1

            if page is not None:
                documents[source]["pages"].add(page)

        results = []

        for document in documents.values():
            results.append(
                {
                    "source": document["source"],
                    "chunks": document["chunks"],
                    "pages": len(document["pages"])
                }
            )

        return sorted(
            results,
            key=lambda item: item["source"].lower()
        )

    def search(
        self,
        query,
        top_k=3
    ):
        if (
            not self.chunks
            or self.embeddings is None
        ):
            return []

        formatted_query = (
            f"query: {query}"
        )

        query_embedding = (
            self.embedding_model.encode(
                [formatted_query],
                normalize_embeddings=True
            )[0]
        )

        scores = (
            self.embeddings
            @ query_embedding
        )

        ranked_indices = (
            scores.argsort()[::-1][:top_k]
        )

        results = []

        for index in ranked_indices:
            results.append(
                {
                    "text":
                        self.chunks[index]["text"],

                    "source":
                        self.chunks[index]["source"],

                    "page":
                        self.chunks[index]["page"],

                    "score":
                        float(scores[index])
                }
            )

        return results