# |**********************************************************************;
# Project           : Why Do We Stop? Textual Explanations for Automated Commentary Driving
#
# Author            : Marc Alexander Kühn, Daniel Omeiza and Lars Kunze
#
# References        : This code is based on the publication and code by Kim et al. [1]
# [1] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata. Textual explanations for self-driving vehicles. In Computer Vision – ECCV 2018, pages 577–593. Springer International Publishing, 2018. doi:10.1007/978-3-030-01216-8_35.
# |**********************************************************************;

from sacrebleu.metrics import BLEU, CHRF, TER
import pickle
from    src.config_VA        import  *
from nltk.translate import meteor_score
from nltk.translate import bleu_score as bleu_nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import json

#config.h5path = "./data/processed_full/"
bleu = BLEU()
chencherry = bleu_nltk.SmoothingFunction()

with open(config.h5path + "extracted_text/" + 'refs_just.pkl', 'rb') as f:
        refs_just = pickle.load(f)
with open(config.h5path + "extracted_text/" + 'refs_desc.pkl', 'rb') as f:
        refs_desc = pickle.load(f)
with open(config.h5path + "extracted_text/" + 'hypo_desc.pkl', 'rb') as f:
        hypo_desc = pickle.load(f)
with open(config.h5path + "extracted_text/" + 'hypo_just.pkl', 'rb') as f:
        hypo_just = pickle.load(f)

# BLEU-4 SacreBleu

#refs = [ # First set of references
#        ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
#        # Second set of references
#        ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.'],
#        ]
#sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

print("SacreBleu")
print("Descriptions BLEU-4:")
score_desc = bleu.corpus_score(hypo_desc, refs_desc)
print(score_desc)

print("Explanations BLEU-4:") #Justifications
score_just = bleu.corpus_score(hypo_just, refs_just)
print(score_just)

print("\n")

# Tokenize each sentence
hypo_desc_tok = []
hypo_just_tok = []
refs_just_tok = []
refs_desc_tok = []
corpus = []
for i in range(len(hypo_desc)):
        corpus.append(refs_desc[0][i])
        corpus.append(refs_just[0][i])
        hypo_desc_tok.append(word_tokenize(hypo_desc[i]))
        hypo_just_tok.append(word_tokenize(hypo_just[i]))
        refs_just_tok.append(word_tokenize(refs_just[0][i]))
        refs_desc_tok.append(word_tokenize(refs_desc[0][i]))
refs_desc_tok = [refs_desc_tok]
refs_just_tok = [refs_just_tok]

#BLEU NLTK

print("BLEU NLTK Corpus Level:")
print("Descriptions BLEU-4:")
score_desc_nltk_corp = bleu_nltk.corpus_bleu(list_of_references=refs_desc_tok[0], hypotheses=hypo_desc_tok) * 100
print(score_desc_nltk_corp)

print("Explanations BLEU-4:") #Justifications
score_just_nltk_corp = bleu_nltk.corpus_bleu(list_of_references=refs_just_tok[0], hypotheses=hypo_just_tok) * 100
print(score_just_nltk_corp)

print("\n")

print("BLEU NLTK Corpus Level with Smoothing:")
print("Descriptions BLEU-4:")
score_desc_nltk_corp = bleu_nltk.corpus_bleu(list_of_references=refs_desc_tok[0], hypotheses=hypo_desc_tok, smoothing_function=chencherry.method7) * 100
print(score_desc_nltk_corp)

print("Explanations BLEU-4:") #Justifications
score_just_nltk_corp = bleu_nltk.corpus_bleu(list_of_references=refs_just_tok[0], hypotheses=hypo_just_tok, smoothing_function=chencherry.method7) * 100
print(score_just_nltk_corp)

print("\n")

print("BLEU NLTK Sentence Level:")
score_desc_m = 0.0
score_just_m = 0.0
for i in range(len(hypo_desc)):
        score_desc_1 = bleu_nltk.sentence_bleu(
                references=word_tokenize(refs_desc[0][i]),
                hypothesis=word_tokenize(hypo_desc[i])
        )
        score_desc_m += score_desc_1


for i in range(len(hypo_just)):
        score_just_1 = bleu_nltk.sentence_bleu(
                references=word_tokenize(refs_just[0][i]),
                hypothesis=word_tokenize(hypo_just[i])
        )
        score_just_m += score_just_1


score_desc_m = (score_desc_m / len(hypo_desc)) * 100
score_just_m = (score_just_m / len(hypo_just)) * 100
print("Descriptions BLEU-4: " + str(score_desc_m))
print("Explanations BLEU-4: " + str(score_just_m))

print("\n")

print("BLEU NLTK Sentence Level with Smoothing:")
score_desc_m = 0.0
score_just_m = 0.0
for i in range(len(hypo_desc)):
        score_desc_1 = bleu_nltk.sentence_bleu(
                references=word_tokenize(refs_desc[0][i]),
                hypothesis=word_tokenize(hypo_desc[i]), smoothing_function=chencherry.method7
        )
        score_desc_m += score_desc_1


for i in range(len(hypo_just)):
        score_just_1 = bleu_nltk.sentence_bleu(
                references=word_tokenize(refs_just[0][i]),
                hypothesis=word_tokenize(hypo_just[i]), smoothing_function=chencherry.method7
        )
        score_just_m += score_just_1


score_desc_m = (score_desc_m / len(hypo_desc)) * 100
score_just_m = (score_just_m / len(hypo_just)) * 100
print("Descriptions BLEU-4: " + str(score_desc_m))
print("Explanations BLEU-4: " + str(score_just_m))

print("\n")

# METEOR # run before once: python -m nltk.downloader all

score_desc_m = 0.0
score_just_m = 0.0
score_desc_ls = []
score_just_ls = []
seq_nr = []
for i in range(len(hypo_desc)):
        score_desc_1 = round(meteor_score.single_meteor_score(
                reference=word_tokenize(refs_desc[0][i]),
                hypothesis=word_tokenize(hypo_desc[i])
        ), 4)
        score_desc_m += score_desc_1
        score_desc_ls.append(str(score_desc_1))
        seq_nr.append(str(i))

for i in range(len(hypo_just)):
        score_just_1 = round(meteor_score.single_meteor_score(
                reference=word_tokenize(refs_just[0][i]),
                hypothesis=word_tokenize(hypo_just[i])
        ), 4)
        score_just_m += score_just_1
        score_just_ls.append(str(score_just_1))

score_desc_m = (score_desc_m / len(hypo_desc)) * 100
score_just_m = (score_just_m / len(hypo_just)) * 100
print("METEOR Descriptions: " + str(score_desc_m))
print("METEOR Explanations: " + str(score_just_m))

#[word_tokenize('The candidate has no alignment to any of the references')],
#                word_tokenize('John loves Mary')

# Save Explanations in file with METEOR Score
with open(config.h5path + "extracted_text/" + "texts.csv", 'w', newline='') as out:
        csv.writer(out, delimiter=';').writerows(
                zip(seq_nr, refs_desc[0], refs_just[0], hypo_desc, hypo_just, score_desc_ls, score_just_ls))

# Write pickles into .json-lists
with open(config.h5path + "extracted_text/" + "refs_desc.json", mode='w', encoding='utf-8') as f:
        feeds = []
        for i in range(len(hypo_desc)):
                entry = {"image_id": str(i), "caption": refs_desc[0][i]}
                feeds.append(entry)
        json.dump(feeds, f)

with open(config.h5path + "extracted_text/" + "refs_just.json", mode='w', encoding='utf-8') as f:
        feeds = []
        for i in range(len(hypo_just)):
                entry = {"image_id": str(i), "caption": refs_just[0][i]}
                feeds.append(entry)
        json.dump(feeds, f)

with open(config.h5path + "extracted_text/" + "hypo_desc.json", mode='w', encoding='utf-8') as f:
        feeds = []
        for i in range(len(hypo_desc)):
                entry = {"image_id": str(i), "caption": hypo_desc[i]}
                feeds.append(entry)
        json.dump(feeds, f)

with open(config.h5path + "extracted_text/" + "hypo_just.json", mode='w', encoding='utf-8') as f:
        feeds = []
        for i in range(len(hypo_just)):
                entry = {"image_id": str(i), "caption": hypo_just[i]}
                feeds.append(entry)
        json.dump(feeds, f)
