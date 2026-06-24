import gradio as gr

from src.rag_pipeline import answer_question, format_sources_markdown


def ask_question(question, product_filter, top_k):
    """
    This function connects the Gradio UI to the RAG pipeline.
    It sends the user's question to the retriever/generator
    and returns both the answer and the retrieved sources.
    """

    if not question or not question.strip():
        return "Please enter a question.", "No sources retrieved."

    result = answer_question(
        question=question,
        top_k=int(top_k),
        product_filter=product_filter
    )

    answer = result["answer"]
    sources = format_sources_markdown(result["sources"])

    return answer, sources


def clear_app():
    """
    Clears the question, answer, and sources.
    """
    return "", "", ""


with gr.Blocks(title="CrediTrust Complaint RAG Chatbot") as demo:
    gr.Markdown(
        """
# CrediTrust Complaint RAG Chatbot

Ask questions about customer complaints across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.

This app retrieves the most relevant complaint excerpts from the full vector store and uses them to generate an evidence-based answer.
"""
    )

    question_input = gr.Textbox(
        label="Ask a question",
        placeholder="Example: Why are customers unhappy with credit cards?",
        lines=3
    )

    with gr.Row():
        product_filter = gr.Dropdown(
            choices=[
                "All",
                "Credit Card",
                "Personal Loan",
                "Savings Account",
                "Money Transfer"
            ],
            value="All",
            label="Product Filter"
        )

        top_k = gr.Slider(
            minimum=3,
            maximum=10,
            value=5,
            step=1,
            label="Number of Retrieved Sources"
        )

    with gr.Row():
        ask_button = gr.Button("Ask")
        clear_button = gr.Button("Clear")

    gr.Markdown("## AI Generated Answer")
    answer_output = gr.Markdown()

    gr.Markdown("## Retrieved Sources")
    sources_output = gr.Markdown()

    ask_button.click(
        fn=ask_question,
        inputs=[question_input, product_filter, top_k],
        outputs=[answer_output, sources_output]
    )

    question_input.submit(
        fn=ask_question,
        inputs=[question_input, product_filter, top_k],
        outputs=[answer_output, sources_output]
    )

    clear_button.click(
        fn=clear_app,
        inputs=[],
        outputs=[question_input, answer_output, sources_output]
    )


if __name__ == "__main__":
    demo.launch()