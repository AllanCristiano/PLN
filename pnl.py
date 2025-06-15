import nltk

# Baixa os recursos necessários, caso ainda não estejam disponíveis.
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

def get_numerical_phrases(sentence):
    """
    Extrai frases numéricas de uma sentença.

    Argumentos:
        sentence: A sentença de entrada.

    Retorna:
        Uma lista de frases numéricas encontradas.
    """
    # Pré-processamento: padroniza o formato de valores monetários
    sentence = sentence.replace('$', ' $').replace('£', ' £').replace('€', ' €')

    # Tokenização
    tokens = nltk.word_tokenize(sentence)
    # Etiquetagem de partes do discurso
    tagged = nltk.pos_tag(tokens)

    # Definição da gramática para frases numéricas
    # Regra 1: Exemplo "more than 5 books"
    # Regra 2: Exemplo "5 books"
    grammar = r"""
        NumericalPhrase: {<NN|NNS>?<RB>?<JJR><IN><CD><NN|NNS>?}
        NumericalPhrase: {<CD><NN|NNS>?}
    """

    # Criação do parser com a gramática
    parser = nltk.RegexpParser(grammar)
    tree = parser.parse(tagged)

    # Extração das frases numéricas
    phrases = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NumericalPhrase'):
        phrases.append(" ".join(word for word, tag in subtree.leaves()))
    return phrases

# Exemplos de uso
sentences = [
    "I have more than 5 books.",
    "The price is less than $10.",
    "The temperature is exactly 25 degrees.",
    "He has at least 3 apples.",
    "She has 2 cats and 3 dogs."
]

for s in sentences:
    print(f"Sentença: {s}")
    print(f"Frases numéricas: {get_numerical_phrases(s)}\n")
