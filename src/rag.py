from src.model_loader import model, tokenizer

def answer_question(question, vectorstore):

    question_lower = question.lower()

    if any(
        phrase in question_lower
        for phrase in [
            "what is this pdf about",
            "summarize",
            "summary",
            "overview",
            "contents",
            "table of contents",
            "topics"
        ]
    ):
        docs = vectorstore.similarity_search(
            "table of contents summary overview topics syllabus",
            k=10
        )
    else:
        docs = vectorstore.similarity_search(
            question,
            k=5
        )

    context_parts = []

    for doc in docs:
        page = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"[Page {page}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": """
You are a PDF question answering assistant.

Answer only using information found in the provided context.

If the answer is not present in the context, reply exactly:

I could not find that information in the PDF.

Do not use outside knowledge.
Do not guess.
Do not make assumptions.

Provide concise and accurate answers.

If the user asks for a summary or overview, provide a structured summary using bullet points.
"""
        },
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{question}
"""
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    pages = sorted(
        {
            doc.metadata.get("page", 0) + 1
            for doc in docs
        }
    )

    response = response.strip()

    if pages:
        response += (
            f"\n\n📄 Sources: Pages "
            f"{', '.join(map(str, pages))}"
        )

    return response