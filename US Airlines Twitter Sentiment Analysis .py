#!/usr/bin/env python
# coding: utf-8

# ### Importing the necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Importing the data

# In[2]:


data=pd.read_csv("Tweets.csv")
data.columns


# In[3]:


data.head()


# In[4]:


data.tail()


# ### Data Understanding and preprocessing

# In[5]:


data.describe(include='all')


# In[6]:


### From the above descriptive statistics we could see that 
### 1.airline_sentiment has most of its tweets negative
### 2.Customer Service Issue is the mostly occured negative reason
### 3.United airlines is the most appeared one among all the airlines.
### 4.Most of the tweets are from Boston MA
### 5.We can see missing values in negativereason,negativereason_confidence,airline_sentiment_gold,negativereason_gold,
###    tweet_coord,tweet_location,user_timezone


# In[7]:


data['negativereason'].value_counts()


# In[8]:


data['tweet_id']


# In[9]:


## checking for null values
data.isnull().sum()


# In[10]:


data.shape


# In[11]:


data['airline_sentiment']


# In[12]:


data['tweet_coord']


# In[16]:


data.info()


# In[17]:


data['airline_sentiment'].value_counts()


# In[18]:


data['airline_sentiment_confidence'].value_counts()


# In[19]:


## number of values with airline_sentiment_confidence<=0.5

(data['airline_sentiment_confidence']<=0.5).value_counts()


# ### Sentiment for each airline
# #### 1.we look for total number of tweets for each airline
# #### 2.Then we will calculate the number of positive,negative,nuetral tweets for each airline
# #### 3.we plot those counts of sentiments for each airline

# In[20]:


print("total number of tweets for each airline \n",data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False))


# In[21]:


#Plotting the number of tweets each airlines has received
colors=sns.color_palette("husl", 10) 
pd.Series(data["airline"]).value_counts().plot(kind = "bar",
                        color=colors,figsize=(8,6),fontsize=10,rot = 0, title = "Total No. of Tweets for each Airlines")
plt.xlabel('Airlines', fontsize=10)
plt.ylabel('No. of Tweets', fontsize=10)


# In[22]:


airlines=['United','US Airways','American','Southwest','Delta','Virgin America']

plt.figure(1,figsize=(15,9)) ## Represents the width and height of overall figure

for i in airlines:
    indices=airlines.index(i) ## will return the indices 0,1,2,3,4,5 for respective values of i
                              ## i.e. for i='United',indices=0 and so on for other values
    
    plt.subplot(2,3,indices+1) ## (x,y,z)it represents the x=height,y=width of plot and z= plot traversal towards right 
                               ## the values should be 1<=num<=6 ,where num is x,y,z
   
    new_df=data[data['airline']==i]
    count=new_df['airline_sentiment'].value_counts()
    Index=[1,2,3]
    plt.bar(Index,count,color=['red','grey','green'])
    plt.xticks(Index,['negative','nuetral','positive'])
    plt.ylabel("count Mood")
    plt.xlabel("Mood")
    plt.title("count of moods of"+i)



# #### The above plots convey us:
# #####  1.United,US Airways and the American airlines are the ones which recieve more negative tweets
# #####  2.VirginAmerica is the airlines with most balanced tweets

# In[23]:


#Plotting the number of each type of sentiments 
colors=sns.color_palette("husl", 10)
pd.Series(data["airline_sentiment"]).value_counts().plot(kind = "bar",
                                    color=colors,figsize=(8,6),rot=0, title = "Total No. of Tweets for Each Sentiment")
plt.xlabel('Sentiments', fontsize=10)
plt.ylabel('No. of Tweets', fontsize=10)


# In[24]:


colors=sns.color_palette("husl", 10)
pd.Series(data["airline_sentiment"]).value_counts().plot(kind="pie",colors=colors,
    labels=["negative", "neutral", "positive"],explode=[0.05,0.02,0.04],
    shadow=True,autopct='%.2f', fontsize=12,figsize=(6, 6),title = "Total Tweets for Each Sentiment")


# ### The most used words in Positive ,Negative and Neutral tweets
# #### 1.wordcloud is the best way to identify the most used words
# #### 2.Wordcloud is a great tool for visualizing nlp data. 
# #### 3.The larger the words in the wordcloud image , the more is the frequency of that word in our text data.

# In[25]:


from wordcloud import WordCloud ,STOPWORDS


# ### wordcloud for negative sentiment of tweets

# In[26]:


new_df=data[data['airline_sentiment']=='negative']
words = ' '.join(new_df['text'])
cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(cleaned_word)
plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[27]:


## In a wordcloud the words which are bigger are the once which occured the most in our given text
## The words which are bigger in the above negative sentiment wordcloud are the ones having more influence among 
## the negative sentimnet texts.
## i.e. Customer service,late flight,cant tell,cancelled flight,plane,help,delay etc.. are most occured and influenced words for negative sentiments.


# In[28]:


#Plotting all the negative reasons 
color=sns.color_palette("husl", 10)
pd.Series(data["negativereason"]).value_counts().plot(kind = "bar",
                        color=color,figsize=(8,6),title = "Total Negative Reasons")
plt.xlabel('Negative Reasons', fontsize=10)
plt.ylabel('No. of Tweets', fontsize=10)


# In[29]:


color=sns.color_palette("husl", 10)
pd.Series(data["negativereason"]).value_counts().head(5).plot(kind="pie",
                labels=["Customer Service Issue", "Late Flight", "Can't Tell","Cancelled Flight","Lost Luggage"],
                colors=color,autopct='%.2f',explode=[0.05,0,0.02,0.03,0.04],shadow=True,
                fontsize=12,figsize=(6, 6),title="Top 5 Negative Reasons")


# ## Word cloud for positive sentiments

# In[30]:


new_df=data[data['airline_sentiment']=='positive']
words = ' '.join(new_df['text'])
cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(cleaned_word)
plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[31]:


## The words which are bigger in the above positive sentiment wordcloud are the ones having more influence among 
## the positive sentiment texts.
## i.e.thank,awesome,great,flight,trip etc.. are most occured and influenced words for positive sentiments.


# ## Word cloud for neutral sentiments

# In[32]:


df=data[data['airline_sentiment']=='neutral']
words = ' '.join(df['text'])
cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(cleaned_word)
plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[33]:


air_senti=pd.crosstab(data.airline, data.airline_sentiment)
air_senti


# In[34]:


percent=air_senti.apply(lambda a: a / a.sum() * 100, axis=1)
percent


# In[35]:


pd.crosstab(index =data["airline"],columns = data["airline_sentiment"]).plot(kind='bar',
                figsize=(10, 6),alpha=0.5,rot=0,stacked=True,title="Airline Sentiment")


# In[36]:


percent.plot(kind='bar',figsize=(10, 6),alpha=0.5,
                rot=0,stacked=True,title="Airline Sentiment Percentage")


# In[37]:


data['tweet_created'] = pd.to_datetime(data['tweet_created'])
data["date_created"] = data["tweet_created"].dt.date


# In[38]:


data["date_created"]


# In[39]:


df = data.groupby(['date_created','airline'])
df = df.airline_sentiment.value_counts()
df.unstack()


# In[40]:


from nltk.corpus import stopwords
stop_words=stopwords.words('english')
print(list(stop_words))


# In[41]:


def tweet_to_words(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 


# In[42]:


def clean_tweet_length(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return(len(meaningful_words)) 


# In[43]:


data['sentiment']=data['airline_sentiment'].apply(lambda x: 0 if x=='negative' else 1)
data.sentiment.head()


# In[44]:


#Splitting the data into train and test
data['clean_tweet']=data['text'].apply(lambda x: tweet_to_words(x))
data['Tweet_length']=data['text'].apply(lambda x: clean_tweet_length(x))
train,test = train_test_split(data,test_size=0.2,random_state=42)


# In[45]:


train_clean_tweet=[]
for tweets in train['clean_tweet']:
    train_clean_tweet.append(tweets)
test_clean_tweet=[]
for tweets in test['clean_tweet']:
    test_clean_tweet.append(tweets)


# In[46]:


from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer(analyzer = "word")
train_features= v.fit_transform(train_clean_tweet)
test_features=v.transform(test_clean_tweet)


# In[51]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# In[52]:


Classifiers = [
    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier()]


# In[53]:



dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['sentiment'])
        pred = fit.predict(test_features)
    except Exception:
        fit = classifier.fit(dense_features,train['sentiment'])
        pred = fit.predict(dense_test)
    accuracy = accuracy_score(pred,test['sentiment'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))  


# In[55]:


Index = [1,2,3,4,5,6]
plt.bar(Index,Accuracy)
plt.xticks(Index, Model, rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Accuracies of Models')

