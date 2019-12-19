# NATURAL LANGUAGE PROCESSING FINAL PROJECT
# December 2019 - Michelle Venables

#IMPORTS:
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import sys
import time
from pprint import pprint
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
np.random.seed(2018)
nltk.download('wordnet')
np.random.seed(0)
from os import path
from googletrans import Translator
from langdetect import detect
from enchant.checker import SpellChecker
import enchant
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
from gensim.utils import lemmatize, simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import pickle
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

#DATA CLEANING:
stop_words = stopwords.words('english')
stop_words.extend(['fly','take','plane','airline','flight','get','unitedairlines', 'unitedairline','unite','make','get','let','tell', 'airlines','from', 'subject', 're', 'edu', 'use', 'not',
                   'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many',
                   'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want',
                   'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
# Import Dataset
data = pd.read_csv('new.csv', index_col=0)
data['tweet'] = data['tweet'].replace(to_replace=r'(http.*)|(https.*)|(pictwitter.*)|(pic.twitter.*)',value=']',regex=True)
data['tweet'] = data['tweet'].replace(to_replace=r'(@)|(#)',value='', regex=True)
#removing emojis1
data = data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))

tweets = data['tweet']
tweets = tweets.map(lambda x: x.lower())

def clean(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. lowercase for all words
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return nopunc
# tweets = tweets.apply(clean)
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation 2. Remove all stopwords 3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(mess)
    stop_words = stopwords.words('english')
    stop_words.extend(['fly','take','plane','airline','flight','get','unitedairlines', 'unitedairline','unite','make','get','let','tell', 'airlines','from', 'subject', 're', 'edu', 'use', 'not',
                   'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many',
                   'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want',
                   'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
    # Now just remove any stopwords
    return [word for word in mess.split() if word.lower() not in stop_words]
def sent_to_words(tweets):
    for twt in tweets:
        twt = re.sub('\s+', ' ', twt)  # remove newline chars
        twt = re.sub("\'", "", twt)  # remove single quotes
        twt = gensim.utils.simple_preprocess(str(twt), deacc=True)
        yield(twt)

@st.cache
def import_data():
# Loading Data
    clean_data = data.tweet.values.tolist()
    data_ready = pickle.load(open('data_ready.pkl', 'rb'))
    dictionary = pickle.load(open('dictionary_sl.pkl', 'rb'))
    lda_model = pickle.load(open('pickles/lda_model.pkl', 'rb'))
    tsne_2d_model = pickle.load(open('tsne2d_model.pkl', 'rb'))
    result = pickle.load(open('result.pkl', 'rb'))
    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]
import_data()


#Calculations
result = pickle.load(open('result.pkl', 'rb'))
result1 = result.loc[result.target_names == "alaskaairlines", "Dominant_Topic"]
result2 = result.loc[result.target_names == "americanairlines", "Dominant_Topic"]
result3 = result.loc[result.target_names == "deltaairlines", "Dominant_Topic"]
result4 = result.loc[result.target_names == "frontierairlines", "Dominant_Topic"]
result5 = result.loc[result.target_names == "jetblueairlines", "Dominant_Topic"]
result6 = result.loc[result.target_names == "southwestairlines", "Dominant_Topic"]
result7 = result.loc[result.target_names == "spiritairlines", "Dominant_Topic"]
result8 = result.loc[result.target_names == "unitedairlines", "Dominant_Topic"]
dom_topics1 = list(result1)
dom_topics2 = list(result2)
dom_topics3 = list(result3)
dom_topics4 = list(result4)
dom_topics5 = list(result5)
dom_topics6 = list(result6)
dom_topics7 = list(result7)
dom_topics8 = list(result8)
top_0_alaska = dom_topics1.count(0.0)
top_1_alaska = dom_topics1.count(1.0)
top_2_alaska = dom_topics1.count(2.0)
top_0_american = dom_topics2.count(0.0)
top_1_american = dom_topics2.count(1.0)
top_2_american = dom_topics2.count(2.0)
top_0_delta = dom_topics3.count(0.0)
top_1_delta = dom_topics3.count(1.0)
top_2_delta = dom_topics3.count(2.0)
top_0_frontier = dom_topics4.count(0.0)
top_1_frontier = dom_topics4.count(1.0)
top_2_frontier = dom_topics4.count(2.0)
top_0_jetblue = dom_topics5.count(0.0)
top_1_jetblue = dom_topics5.count(1.0)
top_2_jetblue = dom_topics5.count(2.0)
top_0_southwest = dom_topics6.count(0.0)
top_1_southwest = dom_topics6.count(1.0)
top_2_southwest = dom_topics6.count(2.0)
top_0_spirit = dom_topics7.count(0.0)
top_1_spirit = dom_topics7.count(1.0)
top_2_spirit = dom_topics7.count(2.0)
top_0_united = dom_topics8.count(0.0)
top_1_united = dom_topics8.count(1.0)
top_2_united = dom_topics8.count(2.0)
total_alaska = top_0_alaska+top_1_alaska+top_2_alaska
total_american = top_0_american+top_1_american+top_2_american
total_delta = top_0_delta+top_1_delta+top_2_delta
total_frontier= top_0_frontier+top_1_frontier+top_2_frontier
total_jetblue= top_0_jetblue+top_1_jetblue+top_2_jetblue
total_southwest= top_0_southwest+top_1_southwest+top_2_southwest
total_spirit= top_0_spirit+top_1_spirit+top_2_spirit
total_united= top_0_united+top_1_united+top_2_united
top_0_per_alaska = round((top_0_alaska/total_alaska)*100,0)
top_1_per_alaska = round((top_1_alaska/total_alaska)*100,0)
top_2_per_alaska = round((top_2_alaska/total_alaska)*100,0)
top_0_per_american = round((top_0_american/total_american)*100,0)
top_1_per_american = round((top_1_american/total_american)*100,0)
top_2_per_american = round((top_2_american/total_american)*100,0)
top_0_per_delta = round((top_0_delta/total_delta)*100,0)
top_1_per_delta = round((top_1_delta/total_delta)*100,0)
top_2_per_delta = round((top_2_delta/total_delta)*100,0)
top_0_per_frontier = round((top_0_frontier/total_frontier)*100,0)
top_1_per_frontier = round((top_1_frontier/total_frontier)*100,0)
top_2_per_frontier = round((top_2_frontier/total_frontier)*100,0)
top_0_per_jetblue= round((top_0_jetblue/total_jetblue)*100,0)
top_1_per_jetblue = round((top_1_jetblue/total_jetblue)*100,0)
top_2_per_jetblue = round((top_2_jetblue/total_jetblue)*100,0)
top_0_per_southwest= round((top_0_southwest/total_southwest)*100,0)
top_1_per_southwest = round((top_1_southwest/total_southwest)*100,0)
top_2_per_southwest = round((top_2_southwest/total_southwest)*100,0)
top_0_per_spirit= round((top_0_spirit/total_spirit)*100,0)
top_1_per_spirit = round((top_1_spirit/total_spirit)*100,0)
top_2_per_spirit = round((top_2_spirit/total_spirit)*100,0)
top_0_per_united= round((top_0_united/total_united)*100,0)
top_1_per_united = round((top_1_united/total_united)*100,0)
top_2_per_united = round((top_2_united/total_united)*100,0)


#APP title
st.title('Airline Twitter Analysis!')
from PIL import Image
ill_img = Image.open("images/illustration-flights-desktop.png")
st.image(ill_img, width=800)
st.markdown('**Overview: **Analyze twitter data of the top 8 domestic airlines and signify areas of improvement!')
st.markdown("**Goals: **")
st.markdown("- Use Topic Modeling in order to classify different areas of interest to twitter users")
st.markdown("- Use sentiment analysis in order to view positive vs negative tweets")
st.markdown("- Show which airlines need more improvement than others.")

visualizations = st.selectbox("Select a visualization", ["","Topic Modeling","t-SNE Clustering & Coherence Graph","Major Airlines", "Low Cost Carriers", "Ultra Low Cost Carriers"])

if visualizations == "Topic Modeling":
    # st.markdown("""<h3 style="text-align: center;"><em><span style="font-size: 24px;">view the topic modeling visualization&nbsp;</span></em></h3>""", unsafe_allow_html=True)
    # st.markdown("""<a href="file:///Users/michellevenables/Documents/CourseMaterials/Projects/Project_6/NLP_finalproject/images/pyLDAvis.html#topic=0&lambda=1&term=">Click Here</a>""", unsafe_allow_html=True)
    cloud_img = Image.open("images/wordcloud.png")
    st.image(cloud_img, width=800)

    weights_img = Image.open("images/topics_weights.png")
    st.image(weights_img, width=800)

if visualizations == "t-SNE Clustering & Coherence Graph":
    #Coherence Scores
    fig = go.Figure(go.Scatter(
        y=[0.32,0.44,0.41,0.39,0.39,0.40,0.42,0.43],
        x=[2,3,4,5,6,7,8,9]))
    fig.update_layout(
        xaxis_title="Number of Topics",
        yaxis_title="Coherence Score",
        title={
            'text': "Coherence Scores",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    #2D t-SNE
    from PIL import Image
    img = Image.open("images/bokeh_plot.png")
    st.image(img, width=600, caption = "2D t-SNE Cluster")
    img2 = Image.open("images/3d_t-SNE.png")
    st.image(img2, width=600, caption = "3D t-SNE Cluster")
    st.plotly_chart(fig)
if visualizations == "Major Airlines":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly

    # Create subplots, using 'domain' type for pie charts
    specs = [[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]]
    fig = make_subplots(rows=2, cols=2, specs=specs)
    labels = ['Bad customer service','Timing and Delays','Great Experience']
    # Define pie charts
    fig.add_trace(go.Pie(labels=labels, values=[top_0_alaska, top_1_alaska, top_2_alaska], name='Alaska'), 1, 1)
    fig.add_trace(go.Pie(labels=labels, values=[top_0_american, top_1_american, top_2_american], name='American'), 1, 2)
    fig.add_trace(go.Pie(labels=labels, values=[top_0_delta, top_1_delta, top_2_delta], name='Delta'), 2, 1)
    fig.add_trace(go.Pie(labels=labels, values=[top_0_united, top_1_united, top_2_united], name='United'), 2, 2)
    fig.update_layout(title={'text': "Major Airlines",'y':0.9,'x':0.48,'xanchor': 'center','yanchor': 'top'}, font=dict(family="Droid Sans", size=18,color="#7f7f7f"))

    st.plotly_chart(fig)

if visualizations == "Low Cost Carriers":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly
# Create subplots, using 'domain' type for pie charts
    specs = [[{'type':'domain'}, {'type':'domain'}]]
    lc_fig = make_subplots(rows=1, cols=2, specs=specs)
    labels = ['Bad customer service','Timing and Delays','Great Experience']
    # Define pie charts

    lc_fig.add_trace(go.Pie(labels=labels, values=[top_0_jetblue, top_1_jetblue, top_2_jetblue], name='Jetblue'), 1, 1)
    lc_fig.add_trace(go.Pie(labels=labels, values=[top_0_southwest, top_1_southwest, top_2_southwest], name='Southwest'), 1, 2)

    lc_fig.update_layout(title={'text': "Low Cost Carrier Airlines",'y':0.9,'x':0.48,'xanchor': 'center','yanchor': 'top'},font=dict(family="Droid Sans", size=18,color="#7f7f7f"))
    st.plotly_chart(lc_fig)

if visualizations == "Ultra Low Cost Carriers":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly

    specs = [[{'type':'domain'}, {'type':'domain'}]]
    ultra_fig = make_subplots(rows=1, cols=2, specs=specs)
    labels = ['Bad customer service','Timing and Delays','Great Experience']
    # Define pie charts

    ultra_fig.add_trace(go.Pie(labels=labels, values=[top_0_frontier, top_1_frontier, top_2_frontier], name='Frontier'), 1, 1)
    ultra_fig.add_trace(go.Pie(labels=labels, values=[top_0_spirit, top_1_spirit, top_2_spirit], name='Spirit'), 1, 2)
    ultra_fig.update_layout(title={'text': "Ultra Low Cost Carriers",'y':0.99,'x':0.48,'xanchor': 'center','yanchor': 'top'},font=dict(family="Droid Sans", size=18,color="#7f7f7f"))

    st.plotly_chart(ultra_fig)

if visualizations == "":
    gifs = st.selectbox("Select a Topic", ["gifs!","Great Experience!","Bad Customer Service...","Delays and Flight Times"])
    if gifs == "Great Experience!":
        st.markdown("""<iframe src="https://giphy.com/embed/2Yd459wzKW9NtkNCAS" width="480" height="356" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/happy-great-cheering-2Yd459wzKW9NtkNCAS"></a></p>""", unsafe_allow_html=True)
        st.markdown("""<iframe src="https://giphy.com/embed/MDM9KWuZukJOg" width="480" height="255" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/dcfilms-MDM9KWuZukJOg"></a></p>""", unsafe_allow_html=True)
    if gifs =="Bad Customer Service...":
        st.markdown("""<iframe src="https://giphy.com/embed/xUOxeZc41DVT2l9laU" width="480" height="271" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/curbyourenthusiasm-season-9-episode-6-xUOxeZc41DVT2l9laU">via GIPHY</a></p>""", unsafe_allow_html=True)
        st.markdown("""<iframe src="https://giphy.com/embed/PznaK2bfR1Hpu" width="480" height="331" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/kristen-wiig-bridesmaids-PznaK2bfR1Hpu"></a></p>""", unsafe_allow_html=True)
        st.markdown("""<<iframe src="https://giphy.com/embed/l1J9u3TZfpmeDLkD6" width="419" height="480" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/angry-mad-anger-l1J9u3TZfpmeDLkD6"></a></p>""", unsafe_allow_html=True)
    if gifs == "Delays and Flight Times":
        st.markdown("""<iframe src="https://giphy.com/embed/9u514UZd57mRhnBCEk" width="480" height="240" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/reaction-9u514UZd57mRhnBCEk">via GIPHY</a></p>""", unsafe_allow_html=True)
