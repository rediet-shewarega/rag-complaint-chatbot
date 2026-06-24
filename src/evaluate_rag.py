from src.rag_pipeline import answer_question


QUESTIONS = [
    "Why are customers unhappy with credit cards?",
    "What are the most common problems in money transfer complaints?",
    "What issues do customers report about personal loans?",
    "What are customers complaining about in savings accounts?",
    "Are customers reporting fraud or unauthorized transactions?",
    "What problems do customers face when resolving disputes?",
    "Which complaints mention delays or slow processing?",
]


def shorten(text, max_chars=250):
    if not text:
        return ""
    text = str(text).replace("\n", " ")
    return text[:max_chars] + "..." if len(text) > max_chars else text


def main():
    output_path = "reports/evaluation_table.md"

    lines = []
    lines.append("# RAG Qualitative Evaluation Table\n")
    lines.append("| Question | Generated Answer | Retrieved Sources | Quality Score (1-5) | Comments/Analysis |")
    lines.append("|---|---|---|---|---|")

    for question in QUESTIONS:
        print(f"Running question: {question}")

        result = answer_question(question, top_k=5)

        answer = shorten(result["answer"], 500)

        source_texts = []

        for source in result["sources"][:2]:
            metadata = source["metadata"]

            product = (
                metadata.get("product_category")
                or metadata.get("product")
                or metadata.get("Product")
                or "N/A"
            )

            issue = (
                metadata.get("issue")
                or metadata.get("Issue")
                or "N/A"
            )

            text = shorten(source["text"], 250)

            source_texts.append(
                f"Source {source['rank']} - Product: {product}; Issue: {issue}; Text: {text}"
            )

        retrieved_sources = "<br>".join(source_texts)

        lines.append(
            f"| {question} | {answer} | {retrieved_sources} | TODO | TODO |"
        )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print(f"\nEvaluation table saved to: {output_path}")
    print("Open the file and replace TODO with your quality score and comments.")


if __name__ == "__main__":
    main()