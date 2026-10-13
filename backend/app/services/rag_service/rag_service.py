
import os
from dotenv import load_dotenv
from google import genai

from app.services.document_service.embeddings import generate_embeddings
from app.services.document_service.vector_store import vector_store

load_dotenv()

class RAGService:
    """Retrieval-Augmented Generation service for KnoVerse AI."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Check your backend .env file."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3.8-flash"

    def answer_question(self, question: str, top_k: int = 3) -> dict:
        """Retrieve relevant document chunks and generate an answer."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Step 1: Convert the question into an embedding.
        query_embedding = generate_embeddings([question])[0]

        # Step 2: Retrieve relevant chunks from FAISS.
        results = vector_store.search(query_embedding, top_k=top_k)

        if not results:
            return {
                "question": question,
                "answer": (
                    "No document information found. "
                    "Please upload a PDF first."
                ),
                "sources": [],
            }

        # Step 3: Build context from retrieved chunks.
        context_parts = []
        sources = []

        for result in results:
            filename = result.get("metadata", {}).get(
                "filename", "Unknown"
            )

            context_parts.append(
                f"Source: {filename}\n"
                f"Content: {result['text']}"
            )

            if filename not in sources:
                sources.append(filename)

        context = "\n\n".join(context_parts)

        # Step 4: Ask Gemini to answer using the retrieved context.
        prompt = f"""
You are KnoVerse AI, a document question-answering assistant.

Answer the user's question using only the document context below.
If the context does not contain the answer, clearly say that
the information was not found in the uploaded documents.
Do not invent facts.

Document context:
{context}

User question:
{question}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return {
            "question": question,
            "answer": response.text or "No answer was generated.",
            "sources": sources,
            "retrieved_chunks": len(results),
        }


rag_service = RAGService()
