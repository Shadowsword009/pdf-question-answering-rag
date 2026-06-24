import gradio as gr

from src.pdf_processor import process_pdf
from src.rag import answer_question

vectorstore = None


def upload_pdf(pdf):
    global vectorstore

    if pdf is None:
        return "Please upload a PDF first."

    try:
        pdf_path = pdf if isinstance(pdf, str) else pdf.name

        vectorstore, chunks = process_pdf(pdf_path)

        return f"✅ PDF processed successfully! Total chunks: {chunks}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def chat(question, history):
    global vectorstore

    if not question.strip():
        return history, history, ""

    if vectorstore is None:
        history.append(
            {
                "role": "assistant",
                "content": "Please upload and process a PDF first."
            }
        )
        return history, history, ""

    answer = answer_question(question, vectorstore)

    history.append(
        {
            "role": "user",
            "content": question
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return history, history, ""


def clear_chat():
    return [], []


with gr.Blocks(title="PDF Question Answering System (RAG)") as demo:

    gr.Markdown(
        """
        # 📄 PDF Question Answering System (RAG)

        Upload a PDF and ask questions about its content.

        **Powered by**
        - Qwen 2.5 1.5B Instruct
        - FAISS
        - LangChain
        - Sentence Transformers
        """
    )

    with gr.Row():

        with gr.Column():

            pdf_file = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )

            process_btn = gr.Button(
                "Process PDF",
                variant="primary"
            )

            status_box = gr.Textbox(
                label="Status",
                interactive=False
            )

        with gr.Column():

            chatbot = gr.Chatbot(
                label="Conversation",
                height=500
            )

            chat_state = gr.State([])

            question_box = gr.Textbox(
                placeholder="Ask a question about the PDF...",
                label="Question"
            )

            with gr.Row():

                ask_btn = gr.Button(
                    "Get Answer",
                    variant="primary"
                )

                clear_btn = gr.Button("Clear")

    process_btn.click(
        fn=upload_pdf,
        inputs=pdf_file,
        outputs=status_box
    )

    ask_btn.click(
        fn=chat,
        inputs=[question_box, chat_state],
        outputs=[chatbot, chat_state, question_box]
    )

    question_box.submit(
        fn=chat,
        inputs=[question_box, chat_state],
        outputs=[chatbot, chat_state, question_box]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, chat_state]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()