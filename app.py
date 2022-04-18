from sentence_transformers import SentenceTransformer

sentence_transformer = SentenceTransformer("bert-base-nli-mean-tokens")

questions = [
    "How do I improve my English speaking? ",
    "How does the ban on 500 and 1000 rupee notes helps to identify black money? ",
    "What should I do to earn money online? ",
    "How can changing 500 and 1000 rupee notes end the black money in India? ",
    "How do I improve my English language? "
]

question_embeddings = sentence_transformer.encode(questions)
print(question_embeddings)

# For german sentences use multilingual mpnet - see: https://www.sbert.net/docs/pretrained_models.html
# sentences = ["This is an example sentence", "Each sentence is converted"]
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
# embeddings = model.encode(sentences)
# print(embeddings)