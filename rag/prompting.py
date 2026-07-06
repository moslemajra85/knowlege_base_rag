SYSTEM_PROMPT = (
    "You are an assistant and an expert in answering questions from chunks of content. "
    "Only answer AI-related questions. Otherwise, say that you cannot answer the question."
)


def build_user_prompt(question, context_chunks):
    retrieved_context = " ".join(context_chunks)

    return (
        "Read the following information that might contain the context you require to answer "
        "the question. You can use the information starting from the <START_OF_CONTEXT> tag "
        "and ending with the <END_OF_CONTEXT> tag. Here is the content:\n\n"
        f"<START_OF_CONTEXT>\n{retrieved_context}\n<END_OF_CONTEXT>\n\n"
        "Please provide an informative and accurate answer to the following question based "
        "on the available context. Be concise and take your time.\n"
        f"Question: {question}\n"
        "Answer:"
    )
