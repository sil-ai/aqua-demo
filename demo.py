import os

import streamlit as st
from allennlp.predictors.predictor import Predictor
import allennlp_models.rc
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer
from scipy import spatial
from laserembeddings import Laser
import plotly.graph_objects as go
import pandas as pd
from readability import Readability

#-------------
# Setup
#-------------

st.title('AQuA Demo 💦')

# Threshold for comparisons between the expected and actual
# comprehensibility answers.
qa_threshold = 0.6

@st.cache(allow_output_mutation=True)
def initialize_models():
    
    # Hugging Face transformers QA model
    hg_comp = pipeline('question-answering')

    # AllenNLP BiDAF model
    allen_comp = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2020.03.19.tar.gz") 

    # Sentence Transformers
    sentence_trans = SentenceTransformer('paraphrase-distilroberta-base-v1')

    return hg_comp, allen_comp, sentence_trans

hg_comp, allen_comp, sentence_trans = initialize_models()

# LASER embeddings

os.system("python -m laserembeddings download-models")

@st.cache(allow_output_mutation=True)
def initialize_laser():
    laser = Laser()
    return laser

laser = initialize_laser()


#------------------
# Comprehensibility
#------------------

st.header('Comprehensibility')

# Context 
context = st.text_area("Candidate Bible Passage (English)", value='', 
        max_chars=None, key=None)

# Question
question = st.text_input('Comprehension Question (English)', value='', 
        max_chars=None, key=None, type='default')

# Expected answer
answer = st.text_input('Expected Answer (English)', value='',
        max_chars=None, key=None, type='default')

# Answer the question
if st.button("Assess Comprehensibility", key=None):
    hg_answer = hg_comp({'question': question, 'context': context})['answer']
    allen_answer = allen_comp.predict(passage=context, question=question)['best_span_str']

    # Compare to expected
    answers = [hg_answer, allen_answer]
    expected = [answer]
    answer_embeddings = sentence_trans.encode(answers)
    expected_embedding = sentence_trans.encode(expected)

    answer_similarities = []
    for emb in answer_embeddings:
        answer_similarities.append(1 - spatial.distance.cosine(emb, 
            expected_embedding[0]))

    score = sum(answer_similarities) / len(answer_similarities)
    if score < qa_threshold:
        
        # Display the wrong answers
        st.error('Probable comprehensibility issue ⚠️  \n\n\n "Transformers" model answer: %s \n\n "BiDAF" model answer: %s' %
                (hg_answer, allen_answer))
    
    else:
        
        # Display the successful answer
        st.balloons()
        st.success('No comprehensibility issue detected ✔️  \n\n\n "Transformers" model answer: %s \n\n "BiDAF" model answer: %s' %
                (hg_answer, allen_answer))


#------------------
# Similarity
#------------------

st.header('Similarity')

# Reference text 
reference = st.text_input("Reference text (English)", value='', 
        max_chars=None, key=None)

# Candidate text
candidate = st.text_input('Candidate text', value='', 
        max_chars=None, key=None, type='default')

# Candidate language
option = st.selectbox('Candidate language',
        ('eng', 'swa', 'tel'))

option_map = {
        'tel': 'te',
        'eng': 'en',
        'swa': 'sw'
        }

option_norm = option_map[option]

# Calculate similarity
if st.button("Assess Similarity", key=None):
    ls_emb = laser.embed_sentences([reference, candidate],
            lang=['en', option_norm])
    similarity = 1 - spatial.distance.cosine(ls_emb[0], ls_emb[1])
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = similarity*100.0,
        title = {'text': "Similarity (%)"},
        gauge = {'axis': {'range': [None, 100]}},
        domain = {'x': [0, 1], 'y': [0, 1]}
        ))
    st.plotly_chart(fig)


#------------------
# Readability
#------------------

st.header('Readability')

# Context 
passage = st.text_area("Candidate Bible Passage (English)", value='', 
        max_chars=None, key='readability_passage')

# Calculate readability
r = Readability(passage)

# Display readability
data = [
        ['Flesch-Kincaid Score', r.flesch_kincaid().score],
        ['Flesch Reading Ease', r.flesch().ease],
        ['Dale Chall Readability Score', r.dale_chall().score],
        ['Automated Readability Index Score', r.ari().score],
        ['Coleman Liau Index', r.coleman_liau().score],
        ['Gunning Fog', r.gunning_fog().score],
        ['Linsear Write', r.linsear_write().score],
        ['Spache Readability Formula', r.spache().score]
        ]
df = pd.DataFrame(data, columns=['Readability Metric', 'Value'])
if st.button('Assess Readability', key=None):
    st.write(df)

