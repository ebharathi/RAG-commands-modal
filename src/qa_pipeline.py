from transformers import pipeline

class QAPipeline:
    def __init__(self, retriever, qa_model_name="deepset/roberta-base-squad2"):
        self.retriever = retriever
        self.qa_pipeline = pipeline("question-answering", model=qa_model_name)

    def answer_question(self, question, top_k=5):
        search_results = self.retriever.search(question, top_k=top_k)
        
        # Combine contexts for more comprehensive answer
        combined_context = " ".join([result["text"] for result in search_results])
        
        answers = []
        for result in search_results:
            # Use individual context for each answer
            answer = self.qa_pipeline(
                question=question, 
                context=result["text"],
                max_answer_len=100  # Adjust as needed
            )
            
            answers.append({
                "answer": answer["answer"],
                "score": answer["score"],
                "files": [result["file"]]  # Use 'files' instead of 'file'
            })
        
        return answers
