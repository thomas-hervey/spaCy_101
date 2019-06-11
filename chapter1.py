from spacy.lang.en import English

nlp = English()

doc = nlp("I like tree kangaroos and narwhals.")

# print the text
print(doc.text)


#
# Documents, Spans, Tokens
#
print('***** Documents, Spans, Tokens *****')

# print the first token
first_token = doc[0]

# print the first token's text
print(first_token.text)

# take a slice of the doc and print
aSlice = doc[2:4]
print(aSlice.text)

# take a larger slice without the end period and print
anotherSlice = doc[2:6]
print(anotherSlice)


#
# Lexical Attributes
#
print('***** Lexical Attributes *****')

doc = nlp(
    "In 1990, more than 60% of people in East Asia were in extreme poverty. "
    "Now less than 4% are."
)

# find percentages

# iterate over the tokens in the doc
for token in doc:
  # check if the token resembles a number
  if token.like_num:
    # get the next token in the doc
    next_token = doc[token.i + 1]

    # check if the next token's text equals '%'
    if next_token.text == "%":
      print("Percentage found:", token.text)


#
# Statistical Models
#

# statistical models enable spaCy to predict linguistic attributes in context (POS, syntactic dependence, NER)
# it's trained on labeled example text; can be updated and tuned
# for example 'en_core_web_sm' has binary weights, vocabulary, meta information (language, pipeline)
print('***** Statistical Models *****')

import spacy

# load the small English model
nlp = spacy.load('en_core_web_sm')

doc = nlp('She ate the pizza')

for token in doc:
  # print the text & predicted POS tag & syntactic dependencies + text
  print(token.text, token.pos_, token.dep_, token.head.text)

  # NOTE: dependency label scheme:
  #         - nsubj (nominal subject): she
  #         - dobj  (direct object): pizza
  #         - det   (determiner article): the

# predicting named entities (NER)
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
  # print the entity text and its label
  print(ent.text, ent.label_)

  # TIP: get quick definitions of the most common tags & labels
  #      by `spacy.explain('GPE')`, `spacy.explain('NNP)`, ...


#
#  (more) Predicting Linguistic Annotation
#
print('***** (more) Predicting Linguistic Annotation *****')


text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"
doc = nlp(text)

for token in doc:
  # get the token text, POS, and tag dependency label
  token_text = token.text
  token_pos = token.pos_
  token_dep = token.dep_

  # print in a formatted way
  print("{:<12}{:<10}{:<10}".format(token_text, token_pos, token_dep))


#
# Predicting named entities in context
#
print('**** Predicting named entities in context ****')

text = "New iPhone X release date leaked as Apple reveals pre-orders by mistake"
doc = nlp(text)

# Iterate over the entities
for ent in doc.ents:
    # Print the entity text and label
    print(ent.text, ent.label_)

# Get the span for "iPhone X"
iphone_x = doc[1:3]

# Print the span text
# NOTE: since the statistical model isn't always right, sometime we have to update the model
print("Missing entity:", iphone_x.text)

#
# ... let's use the matcher
#
print('**** to fix this, lets use the matcher ****')

import spacy

# Import the Matcher
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
doc = nlp("New iPhone X release date leaked as Apple reveals pre-orders by mistake")

# Initialize the Matcher with the shared vocabulary
matcher = Matcher(nlp.vocab)

# Create a pattern matching two tokens: "iPhone" and "X"
pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]

# Add the pattern to the matcher
matcher.add("IPHONE_X_PATTERN", None, pattern)

# Use the matcher on the doc
matches = matcher(doc)
print("Matches:", [doc[start:end].text for match_id, start, end in matches])


#
# one pattern that only matches mentions of the full iOS versions:
# “iOS 7”, “iOS 11” and “iOS 10”.
#
print('**** one pattern that only matches mentions of the full iOS versions: “iOS 7”, “iOS 11” and “iOS 10”.')

import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

doc = nlp(
    "After making the iOS update you won't notice a radical system-wide "
    "redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of "
    "iOS 11's furniture remains the same as in iOS 10. But you will discover "
    "some tweaks once you delve a little deeper."
)

# Write a pattern for full iOS versions ("iOS 7", "iOS 11", "iOS 10")
pattern = [{"TEXT": "iOS"}, {"IS_DIGIT": True}]

# Add the pattern to the matcher and apply the matcher to the doc
matcher.add("IOS_VERSION_PATTERN", None, pattern)
matches = matcher(doc)
print("Total matches found:", len(matches))

# Iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", doc[start:end].text)




#
# Write one pattern that only matches forms of “download”
# (tokens with the lemma “download”), followed by a token
# with the part-of-speech tag 'PROPN' (proper noun).
#
print('**** Write one pattern that only matches forms of “download” (tokens with the lemma “download”), followed by a token with the part-of-speech tag PROPN (proper noun).')

import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

doc = nlp(
    "i downloaded Fortnite on my laptop and can't open the game at all. Help? "
    "so when I was downloading Minecraft, I got the Windows version where it "
    "is the '.zip' folder and I used the default program to unpack it... do "
    "I also need to download Winzip?"
)

# Write a pattern that matches a form of "download" plus proper noun
pattern = [{"LEMMA": "download"}, {"POS": "PROPN"}]

# Add the pattern to the matcher and apply the matcher to the doc
matcher.add("DOWNLOAD_THINGS_PATTERN", None, pattern)
matches = matcher(doc)
print("Total matches found:", len(matches))

# Iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", doc[start:end].text)


#
# one pattern that matches adjectives ('ADJ') followed
# by one or two 'NOUN's (one noun and one optional noun).
#
print('**** one pattern that matches adjectives (ADJ) followed by one or two NOUNs (one noun and one optional noun).')

import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

doc = nlp(
    "Features of the app include a beautiful design, smart search, automatic "
    "labels and optional voice responses."
)

# Write a pattern for adjective plus one or two nouns
pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN", "OP": "?"}]

# Add the pattern to the matcher and apply the matcher to the doc
matcher.add("ADJ_NOUN_PATTERN", None, pattern)
matches = matcher(doc)
print("Total matches found:", len(matches))

# Iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", doc[start:end].text)