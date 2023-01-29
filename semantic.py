import spacy

# medium-sized English model trained on written web text
nlp = spacy.load("en_core_web_md")

#  list of string to compare
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

# print similarity between the words
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# Comparison between "cat" and "monkey" = 0.5929930274321619
# Comparison between "banana" and "monkey" = 0.40415016164997786
# Comparison between "banana" and "cat" = 0.22358825939615987

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + "-", similarity)

    """
    Prints out the sentences next to the similarity to sentence_to_compare."Hello, there is my car" Most similar
    
    Answers:
    where did my dog go- 0.630065230699739
    Hello, there is my car- 0.8033180111627156
    I've lost my car in my car- 0.6787541571030323
    I'd like my boat back- 0.5624939988269558
    I will name my dog Diana- 0.6491444739190607
    """
