"""Generator: Use ChatGroq to answer the question based on retrieved evidence."""
import os
from dotenv import load_dotenv
from groq import Groq

# Use the project key for this app even if the terminal has a stale key set.
load_dotenv(override=True)

# Expects a valid GROQ_API_KEY in the environment or project-root .env.
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_answer(query, context, model="openai/gpt-oss-20b"):
    """Generate a cited answer using ChatGroq."""
    if not context:
        return "I could not find relevant information in the repository to answer your question."

    system_prompt = (
        "You are a strict technical assistant. Answer the user's question using ONLY the provided context blocks. "
        "Each block starts with a citation number like [1]. "
        "You MUST append these citation numbers to the end of any sentence that uses information from that block "
        "(e.g., 'The system uses Chroma for storage [1].'). "
        "If the answer cannot be found in the context, state that explicitly. Do not hallucinate or rely on outside knowledge."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.0,  # Zero temperature forces deterministic, factual outputs
    )
    
    return response.choices[0].message.content
