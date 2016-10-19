#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from pprint import pprint

""" import data """

doc_ko = json.load(open("jawsD.txt"))


""" 1. Data preprocessing """

from konlpy.tag import Twitter
t = Twitter()

def tokenize(doc):
	#norm, stem은 optional
	return['/'.join(t) for t in t.pos(doc,norm=True,stem=True)]
texts_ko = [tokenize(contents)
			for contents in doc_ko['contents_all']]
# tokens = [t for d in texts_ko for t in d]

# pdb.set_trace()
def clean(doc, CleanTerm):
	words_clean = list()
	for sentence in doc:
		Remove = [word for word in sentence \
					if CleanTerm not in word]
		words_clean.append(Remove)

clean_josa, clean_punc = clean(texts_ko, 'Josa'), clean(clean_josa, 'Punctuation')
clean_part = clean(clean_punc, 'KoreanParticle')
texts_clean = clean_part
tokens = [t for d in texts_clean for t in d]

""" 2. Data exploration (feat.NLTK) """

""" morphs가 뭔지 확인하기!!!
# Training data의 token 모으기
tokens_ko = t.morphs(doc_ko)
"""

import nltk
ko = nltk.Text(tokens,name='jawsD')

# Count Tokens
print(len(ko.tokens))	# returns numbers of tokens
print(len(set(ko.tokens)))	#returns number of unique tokens
pprint(ko.vocab().most_common(10))	# returns frequency distribution

# Collocations: 인접하게 빈번하게 등장하는 단어 (예: "text"+"mining")
ko.collocations()
# ko.concordance(u'검색어')
# ko.similar(u'검색어')

""" '작고 노란 ... 짖었다'의 문장구조 그리기 """
# tags_ko = t.pos(unicode('작고 노란 강아지가 고양이에게 짖었다'))
# parser_ko = nltk.RegexpParser("NP:{<Adjective>*<Noun>*}")
# chunks_ko = parser_ko.parse(doc_ko)
# chunks_ko.draw()


# 간단하게 term이 문서에 존재하는지의 유무에 따라 분류해 보자

selected_words = [f[0] for f in ko.vocab().most_common(2000)]	# 최빈도 단어 2000개를 피쳐로 사용

def term_exists(doc):
	return {'exists({})'.format(word):(word in set(doc)) for word in selected_words}
doc_xy = [(term_exists(d),c) for d,c in texts_ko]

""" Topic Modeling && Word Embedding(Word2Vec) """

#encode tokens to integers
# pos = lambda d:['/'.join(p) for p in t.pos(d, stem=True,norm=True)]
# texts_ko = [pos(doc) for doc in doc_ko]


from gensim import corpora

# z = 0
# detoken = list()
# for k in doc_ko['contents_all']:
# 	pos = t.pos(k, norm=True, stem=True)
# 	term = list()
# 	for j in pos:
# 		# detoken[z] = [j[0]]
# 		term.append(j[0])
# 	detoken.append(term)
# 	z += 1

# Create Dictionary by using the tokenized texts
Content = list()
for TokenizedContent in texts_clean:
	word = [TokenizedWord.split('/')[0] \
			for TokenizedWord in TokenizedContent]
	Content.append(word)

# dictionary_ko = corpora.Dictionary(texts_ko)
dictionary_ko = corpora.Dictionary(Content)
dictionary_ko.save('jawsD.dict')

#calulate TF-IDF
from gensim import models
# tf_ko = [dictionary_ko.doc2bow(text) for text in texts_ko]
tf_ko = [dictionary_ko.doc2bow(text) for text in Content]
tfidf_model_ko = models.TfidfModel()
tfidf_ko = tfidf_model_ko[tf_ko]
# corpora.MmCorpus.serialize('jawsD.mm', tfidf_ko)

# Topic model
#LSI
ntopics, nwords = 5, 30
# lsi_ko = models.lsimodel.LsiModel(tfidf_ko, id2word=dictionary_ko, num_topics=ntopics)
lsi_ko = models.lsimodel.LsiModel(tf_ko, id2word=dictionary_ko, num_topics=ntopics)
print(lsi_ko.print_topics(num_topics=ntopics, num_words=nwords))

#LDA
import numpy as np; np.random.seed(42)  # optional
# lda_ko = models.ldamodel.LdaModel(tfidf_ko, id2word=dictionary_ko, num_topics=ntopics)
lda_ko = models.ldamodel.LdaModel(tf_ko, id2word=dictionary_ko, num_topics=ntopics)
print(lda_ko.print_topics(num_topics=ntopics, num_words=nwords))

#HDP
import numpy as np; np.random.seed(42)  # optional
# hdp_ko = models.hdpmodel.HdpModel(tfidf_ko, id2word=dictionary_ko)
hdp_ko = models.hdpmodel.HdpModel(tf_ko, id2word=dictionary_ko)
print(hdp_ko.print_topics(topics=ntopics, topn=nwords))

#Scoring document
# bow = tfidf_model_ko[dictionary_ko.doc2bow(detoken[0])]
bow = [dictionary_ko.doc2bow(Content[0])]
sorted(lsi_ko[bow], key=lambda x: x[1], reverse=True)
sorted(lda_ko[bow], key=lambda x: x[1], reverse=True)
sorted(hdp_ko[bow], key=lambda x: x[1], reverse=True)

# bow = tfidf_model_ko[dictionary_ko.doc2bow(detoken[1])]
bow = [dictionary_ko.doc2bow(Content[1])]
sorted(lsi_ko[bow], key=lambda x: x[1], reverse=True)
sorted(lda_ko[bow], key=lambda x: x[1], reverse=True)
sorted(hdp_ko[bow], key=lambda x: x[1], reverse=True)
