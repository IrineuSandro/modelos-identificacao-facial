import random

def read_words_from_file(filename):
    with open(filename, 'r') as file:
        words = [line.strip() for line in file]
    return words

def generate_name():
    adjectives = read_words_from_file('adjectives.txt')
    nouns = read_words_from_file('nouns.txt')

    adjective = random.choice(adjectives)
    noun = random.choice(nouns)

    tag = f'{adjective}_{noun}'
    return tag
