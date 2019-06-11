######################
# PROCESSING PIPELINES
######################

# build in spaCy pipeline components:
# (POS)        tagger --> Token.tag
# (dependency) parser --> Token.dep, Token.head, Doc.sents, Doc.noun_chunks
#              ner --> Doc.ents, Token.ent_iob, Token.ent_type
# (classifier) textcat --> Doc.cats

# revisit by `print(nlp.pipeline)`

#
# Adding custom components to NLP pipeline
#
print('**** Adding custom components to NLP pipeline')

import spacy

def length_component(doc):
  # Get the doc's length
  doc_length = len(doc)
  print("This document is {} tokens long.".format(doc_length))
  # Return the doc
  return doc

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# add the component first in the pipeline and print the pipe names
nlp.add_pipe(length_component, first=True)
print(nlp.pipe_names)

doc = nlp("this is a sentence")


# Now... let's write a custom component that uses `PhraseMatcher` to find animal names

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

nlp = spacy.load('en_core_web_sm')
animals = ["Golden Retriever", "cat", "turtle", "Rattus norvegicus"]
animal_patterns = list(nlp.pipe(animals))
print("animal paterns: ", animal_patterns)

matcher = PhraseMatcher(nlp.vocab)
matcher.add("ANIMAL", None, *animal_patterns)

# Define the custom component
def animal_component(doc):
  # apply the matcher to the doc
  matches = matcher(doc)

  # create a span for each match and assign the label 'ANIMAL'
  spans = [Span(doc, start, end, label="ANIMAL") for match_id, start, end in matches]

  # overwrite the doc.ents with the matched spans
  doc.ents = spans
  return doc

# add the component to the pipeline after 'ner' component
nlp.add_pipe(animal_component, after="ner")
print(nlp.pipe_names)

# process the text and print the text and label for the doc.ents
doc = nlp("I have a cat and a Golden Retriever")
print([(ent.text, ent.label_) for ent in doc.ents])

#
# using extensions
#
print('**** using extensions')


# use Token.set_extension to register `is_country`, update it for "Spain" and print all tokens

from spacy.lang.en import English
from spacy.tokens import Token

nlp = English()

# Register the Token extension attribute 'is_country' with the default value False
Token.set_extension("is_country", default=False)

# Process the text and set the is_country attribute to True for the token "Spain"
doc = nlp("I live in Spain.")
doc[3]._.is_country = True

# Print the token text and the is_country attribute for all tokens
print([(token.text, token._.is_country) for token in doc])


# now, let's do something to all the tokens, like reverse them

from spacy.lang.en import English
from spacy.tokens import Token

nlp = English()

# Define the getter function that takes a token and returns its reversed text
def get_reversed(token):
    return token.text[::-1]


# Register the Token property extension 'reversed' with the getter get_reversed
Token.set_extension("reversed", getter=get_reversed)

# Process the text and print the reversed attribute for each token
doc = nlp("All generalizations are false, including this one.")
for token in doc:
    print("reversed:", token._.reversed)


#
# Extensions for spans on entities
#
print('**** Extensions for spans on entities')

# let's get wikipedia url's for people and places given some text

import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")


def get_wikipedia_url(span):
    # Get a Wikipedia URL if the span has one of the labels
    if span.label_ in ("PERSON", "ORG", "GPE", "LOCATION"):
        entity_text = span.text.replace(" ", "_")
        return "https://en.wikipedia.org/w/index.php?search=" + entity_text


# Set the Span extension wikipedia_url using get getter get_wikipedia_url
Span.set_extension("wikipedia_url", getter=get_wikipedia_url)

doc = nlp(
     "In over fifty years from his very first recordings right through to his "
    "last album, David Bowie was at the vanguard of contemporary culture."
  	"He lived in Great Britain for most of his life."
)
for ent in doc.ents:
    # Print the text and Wikipedia URL of the entity
    print(ent.text, ent._.wikipedia_url)


#
# now link components with extensions
#
print('**** now link components with extensions; NOTE: THIS WONT WORK WITHOUT THE DATA')

# let's get capitals of countries from text

import json
from spacy.lang.en import English
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher

with open("exercises/countries.json") as f:
    COUNTRIES = json.loads(f.read())

with open("exercises/capitals.json") as f:
    CAPITALS = json.loads(f.read())

nlp = English()
matcher = PhraseMatcher(nlp.vocab)
matcher.add("COUNTRY", None, *list(nlp.pipe(COUNTRIES)))


def countries_component(doc):
    # Create an entity Span with the label 'GPE' for all matches
    matches = matcher(doc)
    doc.ents = [Span(doc, start, end, label="GPE") for match_id, start, end in matches]
    return doc


# Add the component to the pipeline
nlp.add_pipe(countries_component)
print(nlp.pipe_names)

# Getter that looks up the span text in the dictionary of country capitals
get_capital = lambda span: CAPITALS.get(span.text)

# Register the Span extension attribute 'capital' with the getter get_capital
Span.set_extension("capital", getter=get_capital)

# Process the text and print the entity text, label and capital attributes
doc = nlp("Czech Republic may help Slovakia protect its airspace")
print([(ent.text, ent.label_, ent._.capital) for ent in doc.ents])

#
# More efficient text processing
#
print('**** More efficient text processing')

# we can run only certain parts, like the tokenizer by `nlp.make_doc`
# we can also add "context", by attaching metadata to a doc by setting `as_tuples=True` on `nlp.pipe`
# this lets us pass in (text, context)
# lastly, we can disable pipeline components by `with nlp.disable_pipes('tagger'):`

# let's run a pipeline to iterate many texts. Let's use nlp.pipe
# NOTE: This won't work without the file
import json
import spacy

nlp = spacy.load("en_core_web_sm")

with open("exercises/tweets.json") as f:
    TEXTS = json.loads(f.read())

# Process the texts and print the adjectives
for doc in nlp.pipe(TEXTS):
    print([token.text for token in doc if token.pos_ == "ADJ"])


# or create docs in a list

import json
import spacy

nlp = spacy.load("en_core_web_sm")

with open("exercises/tweets.json") as f:
    TEXTS = json.loads(f.read())

# Process the texts and print the entities
docs = list(nlp.pipe(TEXTS))
entities = [doc.ents for doc in docs]
print(*entities)