from transformers import pipeline

class QAPipeline:
    def __init__(self, retriever, qa_model_name="deepset/roberta-base-squad2"):
        self.retriever = retriever
        self.qa_pipeline = pipeline("question-answering", model=qa_model_name)

    def answer_question(self, question, top_k=5):
        search_results = self.retriever.search(question, top_k=top_k)
        answers = []

        for result in search_results:
            context = result["text"]
            answer = self.qa_pipeline(question=question, context=context)
            answers.append({
                "answer": answer["answer"],
                "score": answer["score"],
                "file": result["file"]
            })

        return answers
