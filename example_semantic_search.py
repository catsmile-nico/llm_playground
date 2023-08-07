
def word_embeddings():
# Word embeddings are like special codes for words that help computers understand which words are friends and should be close together
    from gensim.models import Word2Vec

    # Let's make some sentences to play with
    sentences = [
        ["I", "love", "ice", "cream"],
        ["Ice", "cream", "is", "delicious"],
        ["I", "scream", "for", "ice", "cream"]
    ]

    # Let's train our word embeddings toy on these sentences
    model = Word2Vec(sentences, vector_size=10, window=3, min_count=1, sg=0)

    # Now, let's see what our toy has learned about words!
    similar_words = model.wv.most_similar("cream", topn=2)

    print("Words similar to 'ice':", similar_words)

def part_of_speech():
    # part-of-speech helps us understand the different roles of words in sentences
    # - Nouns: These are like the main characters in the story. They can be people, animals, places, or things. For example, "dog," "cat," "beach," and "toy."
    # - Verbs: These are the action words. They show what the characters are doing. For example, "run," "jump," "sing," and "play."
    # - Adjectives: These are like describing words. They tell us more about the characters or things. For example, "big," "happy," "red," and "shiny."
    # - Adverbs: These are like how characters do things. They describe how an action happens. For example, "quickly," "happily," "loudly," and "carefully."
    # - Prepositions: These show the relationships between things. They tell us where or when something happens. For example, "on," "under," "in," and "before."
    # - Conjunctions: These are like glue words. They connect words or sentences together. For example, "and," "but," "because," and "or."
    # - Pronouns: These are like shortcuts for nouns. Instead of saying a name all the time, we use pronouns. For example, "he," "she," "it," and "they."

    import spacy

    # Let's call our magical friend
    nlp = spacy.load("en_core_web_sm")

    # Our story
    story = "The curious cat chased the playful dog through the green garden."

    # Let's ask our magical friend to analyze the story
    doc = nlp(story)

    # Now, let's see the roles of each word
    for word in doc:
        print(word.text, "-", word.pos_)

def bag_of_words():
    # learn about the text by counting the words in its bag

    from sklearn.feature_extraction.text import CountVectorizer

    # Let's create our bag-of-words tool
    vectorizer = CountVectorizer()

    # Our stories or sentences
    stories = [
        "The cat chased the mouse.",
        "The dog played in the park.",
        "The mouse squeaked and ran away."
    ]

    # Let's use the tool to count the words
    word_counts = vectorizer.fit_transform(stories)

    # Now, we can see the words and their counts
    print("Words and their counts:")
    print(vectorizer.get_feature_names_out())
    print(word_counts.toarray())

def sliding_window():
    import spacy
    from sklearn.feature_extraction.text import CountVectorizer

    # Let's call our magical friend for parts-of-speech
    nlp = spacy.load("en_core_web_sm")

    # Let's create our bag-of-words tool
    vectorizer = CountVectorizer()

    # Our text excerpt
    text = "The brave knight fought the dragon in the dark cave. The dragon roared and breathed fire."

    # Split the text into sentences
    sentences = text.split(". ")

    # Create a sliding window of size 3 words
    window_size = 3

    # Let's process each sentence with our magical friend and sliding window
    word_groups = []
    for sentence in sentences:
        doc = nlp(sentence)
        words = [token.text for token in doc]
        for i in range(len(words) - window_size + 1):
            word_group = " ".join(words[i:i+window_size])
            word_groups.append(word_group)

    # Let's use the tool to count the word groups
    word_group_counts = vectorizer.fit_transform(word_groups)

    # Now, we can see the word groups and their counts
    print("Word groups and their counts:")
    print(vectorizer.get_feature_names_out())
    print(word_group_counts.toarray())

# %%
def find_sentence_from_excerpts():
    import spacy
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Input sentence
    input_sentence = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."

    # List of data sentences
    data_sentences = [
        "1) as he chooses. nobody wants him to come. though i shall always say that he used my daughter extremely ill ; and if i was her, i",
        "2) is such a man ! '' `` yes, yes, they must marry. there is nothing else to be done. but there are two things that i",
        "3) he is such a man ! '' `` yes, yes, they must marry. there is nothing else to be done. but there are two things that",
        "4) he chooses. nobody wants him to come. though i shall always say that he used my daughter extremely ill ; and if i was her, i would",
        "5) , that a single man in possession of a good fortune, must be in want of a wife. however little known the feelings or views of such a",
        "6) acknowledged, that a single man in possession of a good fortune, must be in want of a wife. however little known the feelings or views of such",
        "7) a man ! '' `` yes, yes, they must marry. there is nothing else to be done. but there are two things that i want very",
        "8) man ! '' `` yes, yes, they must marry. there is nothing else to be done. but there are two things that i want very much",
        "9) chooses. nobody wants him to come. though i shall always say that he used my daughter extremely ill ; and if i was her, i would not",
        "10) it, '' said her daughter, `` she would go. '' `` she is a very fine-looking woman ! and her calling here was prodigiously civil ! for",
        "11) '' said her daughter, `` she would go. '' `` she is a very fine-looking woman ! and her calling here was prodigiously civil ! for she only",
        "12) , '' said her daughter, `` she would go. '' `` she is a very fine-looking woman ! and her calling here was prodigiously civil ! for she",
        "13) said her daughter, `` she would go. '' `` she is a very fine-looking woman ! and her calling here was prodigiously civil ! for she only came",
        "14) perpetually talked of. my mother means well ; but she does not know, no one can know how much i suffer from what she says. happy shall",
        "15) that a single man in possession of a good fortune, must be in want of a wife. however little known the feelings or views of such a man",
        "16) such a man ! '' `` yes, yes, they must marry. there is nothing else to be done. but there are two things that i want",
        "17) a single man in possession of a good fortune, must be in want of a wife. however little known the feelings or views of such a man may",
        "18) room, `` and what do you think of my husband ? is not he a charming man ? i am sure my sisters must all envy me. i",
        "19) , well ! it is just as he chooses. nobody wants him to come. though i shall always say that he used my daughter extremely ill ; and",
        "20) every thing else she is as good natured a girl as ever lived. i will go directly to mr. bennet, and we shall very soon settle it with",
    ]

    # Calculate sentence embeddings using spaCy
    input_embedding = nlp(input_sentence).vector
    data_embeddings = [nlp(sentence).vector for sentence in data_sentences]

    # Calculate cosine similarity scores
    similarity_scores = cosine_similarity([input_embedding], data_embeddings)[0]

    # Find the index of the most similar data sentence
    most_similar_index = np.argmax(similarity_scores)

    # Print the most similar data sentence
    print("Input Sentence:", input_sentence)
    print("Most Similar Data Sentence:", data_sentences[most_similar_index])

# %%
find_sentence_from_excerpts()

# %%
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster

def merge_overlapping_results(data_sentences, similarity_threshold=0.4):
    """From a list of sentences, group them into clusters based on their similarity

    Args:
        data_sentences (list): List of sentences
        similarity_threshold (float, optional): Threshold to group sentences. Lower the stricter. Defaults to 0.4.

    Returns:
        Dictionary: A dictionary of clusters and their sentences.
    """

    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Calculate sentence embeddings using spaCy
    data_embeddings = [nlp(sentence).vector for sentence in data_sentences]

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(data_embeddings)

    # Perform hierarchical clustering
    linkage_matrix = linkage(similarity_matrix, method='complete')

    # Apply clustering with a similarity threshold
    cluster_labels = fcluster(linkage_matrix, t=similarity_threshold, criterion='distance')

    # Initialize a dictionary to store merged results for each cluster
    merged_results = {}

    # Group sentences by cluster label
    for idx, cluster_label in enumerate(cluster_labels):
        if cluster_label not in merged_results:
            merged_results[cluster_label] = [data_sentences[idx]]
        else:
            merged_results[cluster_label].append(data_sentences[idx])

    return list(merged_results.values())

# Example data sentences
data_sentences = [
    "1) as he chooses. nobody wants him to come. though i shall always say that he used my daughter extremely ill ; and if i was her, i",
    "2) is such a man ! '' `` yes, yes, they must marry. there is nothing else to be done. but there are two things that i",
    "3) he is such a man ! '' `` yes, yes, they must marry. there is nothing else to be done. but there are two things that",
    "4) he chooses. nobody wants him to come. though i shall always say that he used my daughter extremely ill ; and if i was her, i would",
    "5) , that a single man in possession of a good fortune, must be in want of a wife. however little known the feelings or views of such a",
    "6) acknowledged, that a single man in possession of a good fortune, must be in want of a wife. however little known the feelings or views of such",
    "7) a man ! '' `` yes, yes, they must marry. there is nothing else to be done. but there are two things that i want very",
    "8) man ! '' `` yes, yes, they must marry. there is nothing else to be done. but there are two things that i want very much",
    "9) chooses. nobody wants him to come. though i shall always say that he used my daughter extremely ill ; and if i was her, i would not",
    "10) it, '' said her daughter, `` she would go. '' `` she is a very fine-looking woman ! and her calling here was prodigiously civil ! for",
    "11) '' said her daughter, `` she would go. '' `` she is a very fine-looking woman ! and her calling here was prodigiously civil ! for she only",
    "12) , '' said her daughter, `` she would go. '' `` she is a very fine-looking woman ! and her calling here was prodigiously civil ! for she",
    "13) said her daughter, `` she would go. '' `` she is a very fine-looking woman ! and her calling here was prodigiously civil ! for she only came",
    "14) perpetually talked of. my mother means well ; but she does not know, no one can know how much i suffer from what she says. happy shall",
    "15) that a single man in possession of a good fortune, must be in want of a wife. however little known the feelings or views of such a man",
    "16) such a man ! '' `` yes, yes, they must marry. there is nothing else to be done. but there are two things that i want",
    "17) a single man in possession of a good fortune, must be in want of a wife. however little known the feelings or views of such a man may",
    "18) room, `` and what do you think of my husband ? is not he a charming man ? i am sure my sisters must all envy me. i",
    "19) , well ! it is just as he chooses. nobody wants him to come. though i shall always say that he used my daughter extremely ill ; and",
    "20) every thing else she is as good natured a girl as ever lived. i will go directly to mr. bennet, and we shall very soon settle it with",
]

# Merge overlapping results
merged_results = merge_overlapping_results(data_sentences)

# Print merged results
for idx, cluster in enumerate(merged_results, start=1):
    print(f"Cluster {idx}:")
    print("\n".join(cluster))
    print("="*50)

# %%
