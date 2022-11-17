
'''
OpenAI Synthetic Language Task:

C. Building Blocks for Synthetic Tasks
We describe the 13 building blocks used to create synthetic tasks for evaluating model performance as a function of docstring complexity. Each building block is specified by a line of text and a line of code:

'''
from collections import namedtuple

Element = namedtuple('Element', ['description', 'rule'])

synthetic_grammar = dict(
    remove_all=Element(
        "remove all instances of the letter e from the string",
        lambda s : s.replace("e", "")),
    replace_spaces=Element(
        "replace all spaces with exclamation points in the string",
        lambda s : s.replace(" ", "!")),
    lower=Element(
        "convert the string s to lowercase",
        lambda s : s.lower()),
    remove_first_two_chars=Element(
        "remove the first and last two characters of the string",
        lambda s : s[2:-2]),
    remove_vowels=Element(
        "removes all vowels from the string",
        lambda s : "".join(char for char in s if char not in "aeiouAEIOU")),
    remove_every_third=Element(
        "remove every third character from the string",
        lambda s : "".join(char for i, char in enumerate(s) if i % 3 != 0)),
    drop_last_half=Element(
        "drop the last half of the string, as computed by characters",
        lambda s : s[: len(s) // 2]),
    replace_spaces_with_triple_space=Element(
        "replace spaces with triple spaces",
        lambda s : s.replace(" ", "   ")),
    reverse_word_order=Element(
        "reverse the order of words in the string",
        lambda s : " ".join(s.split()[::-1])),
    drop_first_half_by_words=Element(
        "drop the first half of the string, as computed by number of words",
        lambda s : " ".join(s.split()[len(s.split ()) // 2 :])),
    add_apples_after_each_words=Element(
        "add the word apples after every word in the string",
        lambda s : " ".join(word + " apples" for word in s.split())),
    every_other_uppercase=Element(
        "make every other character in the string uppercase",
        lambda s : "".join(char.upper() if i % 2 == 0 else char for i, char in enumerate(s))),
    delete_punctuation=Element(
        "delete all excamation points, question marks, and periods from the string",
        lambda s : "".join([x for x in s if x not in ".!?"])))

''' This synthetic task is arbitrary, and a lot of the operators could be generalized into a grammar '''

def compose(a, b, connective='then'):
    ''' [Element] -> Element '''
    return Element(a.description + ' ' + connective + ' ' + b.description,
                   lambda s : a.rule(b.rule(s)))

def generate(depth=3):
    ''' Generate without pruning? '''
    generated = list(synthetic_grammar.values()) # depth 1
    for i in range(depth - 1):
        new = [compose(el, new_el) for el in generated
               for new_el in synthetic_grammar.values()
               if new_el.description not in el.description]
        generated += new
    return generated

import string
test_cases=dict(
    lorem='''Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum?''',
    foxdog='''The quick brown fox jumps over a lazy dog!''',
    zebra='How quickly daft jumping zebras vex!',
    punctuation=string.punctuation,
    hexdigits=string.hexdigits,
    whitespace=string.whitespace
    )

def main():
    for el in generate(depth=2):
        print(el)

if __name__ == '__main__':
    main()
