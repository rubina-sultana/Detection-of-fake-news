# Detection-of-fake-news
<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Project Title

Final project for the Building AI course

## Summary

Due to increasing use of internet ,it is now easy to spread fake news . A huge number of persons are regularly connected with internet and social media platorms. There is no any restriction while posting any news on these platforms . So some of the people takes the advantage of these platforms and start spreading the fake news against the individuals and organizations . This  can destroy the repute of an individual or can effect a business.Through fake news , the opinions of the people can also been changed.
        In this systematic literature review, the supervised machine learning classifiers are discussed that requires the labeled data for mining . In future a research can be on the use of the unsupervised machine learning classifiers for the detection of fake news.


## Background

Which problems does your idea solve?
 * Fake news is frequently used to target minorities and has become a significant cause of localised violence as well as large scale riots.
 * Engineered mass violence was instigated during the 2013 ,though a disinformation campaign propagating the love jihad conspiracy theory and circulationg  a fake news         video.
How common or frequent is this problem?
About three-quarters of Americans who say they follow news and currents events agree that fake news is a big problem today , according to findings from Delloite's recent Digital Media Trends. 

What is your personal motivation?  
I gone through many  websites and blogs which spread fake news especially in jobs . they'll demand some money regarding job applications.Later ther'll be no response.
this occurs mostly in countries facing unemployment.

Why is this topic important or interesting?
Research involves many facts why people wants to spread fake news and who are encouraging them to do so. Finally  by using some techniques in Artificial Intelligence we stop fake news problems many websites.

Main problem areas:
*  Main problem is many organizations do not provide information for analysis for this we need permissions from various designations.
*   proper address of fake news posting users are not available in websites.


## How is it used?

In one of earlier works, Karimi et al. ] use convolutional neural network (CNN) and LSTM methods to combine various text-based features, such as those from statements (claims) related to news data. Liu et al.  also use RNN and CNN-based methods to build propagation paths for detecting fake news at the early stage of its propagation. Shu et al. propose a matrix factorization method TriFN to model the relationships among the publishers, news stories and social media users for fake news detection.


![Fake news](https://www.bing.com/images/search?view=detailV2&ccid=SoeOCWGq&id=5119CD8BB91C75885410B3EC10A3875D3B22E070&thid=OIP.SoeOCWGq_T3a-XotRb2isgHaE8&mediaurl=https%3a%2f%2fmedia.istockphoto.com%2fphotos%2ffake-news-computer-keyboard-with-fake-news-key-enlarged-by-a-glass-picture-id863785500%3fk%3d6%26m%3d863785500%26s%3d612x612%26w%3d0%26h%3dkVzSPzFrR4acCcql0XJe6stDQNJv8ZzF-QJXLF_3fz4%3d&exph=408&expw=612&q=fake+news+images&simid=608036871301694333&FORM=IRPRST&ck=0CE54ECBA35F6777C8DAD7F2C39AEA21&selectedIndex=48)

If you need to resize images, you have to use an HTML tag, like this:
<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg" width="300">

Example code:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt      /import the downloaded data set/
data = pd.read_csv('News.csv',index_col=0)
data.head()
data.shape /data preporocessing/
data = data.drop(["title", "subject","date"], axis = 1)
data.isnull().sum()  /so thers is no null value/
# Shuffling
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)
sns.countplot(data=data,
			x='class',
			order=data['class'].value_counts().index)
from tqdm import tqdm
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud    /Once we have all the required modules, we can create a function name preprocess text. This function will preprocess all the data given as input./
Real
consolidated = ' '.join(
    word for word in data['text'][data['class'] == 1].astype(str))
wordCloud = WordCloud(width=1600,
                      height=800,
                      random_state=21,
                      max_font_size=110,
                      collocations=False)
plt.figure(figsize=(15, 10))
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.axis('off')
plt.show()


![image](https://user-images.githubusercontent.com/129534370/229294973-c712131a-5efa-415b-b53b-b6e965dff887.png)



## Data sources and AI methods
Source The source of news (e.g. CNN, BBC).



| Syntax      | Description |
| ----------- | ----------- |
| FNaD     | SSDs      |
| it develops a new theoretical framework suited to mitigate the impacts of FNaD on SCDs and it analyses the relationship using a specific dataset and support-vector machine.   

## Challenges

1.Change in Public Opinion
2.Defamation is among the disadvantages of fake news
3. Fake News may lead to Social Unrest
4. Fake News Cost lives
## What next?

1.Assess trained model performance
2.Build and train the model
3.import libraries and datasets
## Acknowledgments

https://www.bing.com/ck/a?!&&p=b8ba8fd224610be0JmltdHM9MTY4MDMwNzIwMCZpZ3VpZD0wOTVlMTAxYi1kZmU3LTY0MzYtMjU1Yi0wMThkZGU3YzY1NTEmaW5zaWQ9NTIxMA&ptn=3&hsh=3&fclid=095e101b-dfe7-6436-255b-018dde7c6551&psq=program+for+detction+of+fake+news+in+ai&u=a1aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmcvZmFrZS1uZXdzLWRldGVjdGlvbi11c2luZy1tYWNoaW5lLWxlYXJuaW5nLw&ntb=1

