from retriever import load_and_index_data
from qa_pipeline import QAPipeline

def main():
    # Load and index documents
    data_folder = "./data"
    retriever = load_and_index_data(data_folder)
    
    # Initialize QA pipeline
    qa_pipeline = QAPipeline(retriever)
    
    # Ask questions
    print("AI Terminal Assistant")
    print("Ask your question (type 'exit' to quit):")
    while True:
        question = input("> ")
        if question.lower() == "exit":
            break
        
        answers = qa_pipeline.answer_question(question, top_k=3)
        print("\nResults:")
        for answer in answers:
            print(f"File: {answer['file']}")
            print(f"Answer: {answer['answer']}")
            print(f"Confidence: {answer['score']:.2f}")
            print("-" * 40)

if __name__ == "__main__":
    main()
