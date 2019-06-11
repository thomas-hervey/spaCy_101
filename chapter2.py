#######################################
# LARGE-SCALE DATA ANALAYSIS WITH spaCy
#######################################

# shared vocab and string store
# `vocab` stores data shared across multiple documents
# strings are only stored once in the `stringStore` via `nlp.vocab.strings`
# a `Lexeme` object is an entry in the vocabulary, containing context-independent info about a word

#
# Strings to hashes
#
print('*** Strings to hashes')

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I have a cat")

# look up the hash for the word 'cat'
cat_hash = nlp.vocab.strings['cat']
print(cat_hash)

# look up the cat_hash to get the string
cat_string = nlp.vocab.strings[cat_hash]
print(cat_string)

#
# Creating docs from scratch
#
print('**** Creating docs from scratch')

import spacy

nlp = spacy.load("en_core_web_sm")

# import the Doc class
from spacy.tokens import Doc

# Desired text: "spaCy is cool!"
words = ["spaCy", "is", "cool", "!"]
spaces = [True, True, False, False]

# create a doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)

#
# Docs, spans, and entities from scratch
#
print('**** Docs, spans, and entities from scratch')

from spacy.lang.en import English

nlp = English()

# Import the Doc and Span classes
from spacy.tokens import Doc, Span

words = ["I", "like", "David", "Bowie"]
spaces = [True, True, True, False]

# Create a doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)

# Create a span for "David Bowie" from the doc and assign it the label "PERSON"
span = Span(doc, 2, 4, label="PERSON")
print(span.text, span.label_)

# Add the span to the doc's entities
doc.ents = [span]

# Print entities' text and labels
print([(ent.text, ent.label_) for ent in doc.ents])


#
# Using native tokens, find proper nouns before verbs
#
print('**** Using native tokens, find proper nouns before verbs')

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Berlin is a nice city")

for token in doc:
  if token.pos_ == 'PROPN':
    # check if the next token is a verb
    if doc[token.i + 1].pos_ == 'VERB':
      print('Found proper noun before a verb:', token.text)


#
# Word vectors and semantic similarity
#
print('**** Word vectors and semantic similarity')

# spaCy can compare two objects and predict similarity
# `Doc.similarity()`, `Span.similarity()`, `Token.similarity()`
# needs medium or larger model
# NOTE: This is done using word vectors
# (default cosine similarity, but can be adjusted)

# inspecting word vectors

import spacy

# Load the en_core_web_md model
nlp = spacy.load("en_core_web_md")

# Process a text
doc = nlp("Two bananas in pyjamas")

bananas_vector = doc[1].vector
print(bananas_vector)


# comparing similarities

import spacy

nlp = spacy.load("en_core_web_md")

doc1 = nlp("It's a warm summer day")
doc2 = nlp("It's sunny outside")

# Get the similarity of doc1 and doc2
similarity = doc1.similarity(doc2)
print(similarity)


#
# Example of phrase matching
#
print('**** example of phrase matching (finding countries from a text); NOTE: WONT RUN WITHOUT FILES')

from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
import json

with open("exercises/countries.json") as f:
    COUNTRIES = json.loads(f.read())
with open("exercises/country_text.txt") as f:
    TEXT = f.read()

nlp = English()
matcher = PhraseMatcher(nlp.vocab)
patterns = list(nlp.pipe(COUNTRIES))
matcher.add("COUNTRY", None, *patterns)

# Create a doc and find matches in it
doc = nlp(TEXT)

# Iterate over the matches
for match_id, start, end in matcher(doc):
    # Create a Span with the label for "GPE"
    span = Span(doc, start, end, label="GPE")

    # Overwrite the doc.ents and add the span
    doc.ents = list(doc.ents) + [span]

    # Get the span's root head token
    span_root_head = span.root.head
    # Print the text of the span root's head token and the span text
    print(span_root_head.text, "-->", span.text)

# Print the entities in the document
print([(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "GPE"])
