# Fake_News_detection

Machine learning Project, which uses logistic regression  and SVM method to  classify the news into FAKE/REAL, on the basis of their Title and Body-Content.
Data has been collected from 3 different sources.





The codeFiles folder having below files in FetchindData Folder
## First step is to fetch data.

FetchingNYTdata.py -> code for scrapping data from New York Times
FetchingGuardianData.py -> code for scrapping data from Guradians Post newws.

## Second step is to execute files in Cleaning data folder.

Cleaning data Kaggle.py -> code for cleaning Fake news data

CleaningGuardianDataset.py -> code for cleaning Guardian post data 

CleaningNYTDataset.py -> code for cleaning New York Times data

After cleaning all the three data, we will get kaggle_cleaned.csv, NYT_cleaned.csv and guardian_cleaned.csv files.

## Third step is to execute file from Combining and Execution folder.


CombiningDataset.py -> Code for implementation of models - Logistic regression
To execute the code, First step is to scrape the data from NYT and guardians Files. To execute these run FetchingNYTdata.py and FetchingGuardianData.py After scraping NYT data, we will get data into mongodb collection and export using mongoexport into NYT_DB.csv file. After scraping data from Guradian post, we will get data in .json files.

Execute CombiningDataset.py file for combining the three datasets and training the models. The combined datasets are in file Final.csv file. This csv file will be input for training the classifiers.

## The attributes of the final dataset are: 
- Id: A unique id for each new article 
- Publication date: News article publication date 
- Headline: Headline of the news article 
- Body: The  content of the article 
- Fakeness: binary value 0/1 for real and fake news respectively. 

## Libraries used

- BeautifulSoup – a python library to for pulling data out of HTML and XML  files. To scrape or extract all the data from a particular website
- pymongo - To establish connection between python code and mongo  is used to retrieve the data from MongoDB and store locally.
- pandas - for reading the csv and converting it to dataframe
- sklearn- scikit learn for shuffling the data, performing cross validation, performance  metrics
- seaborn & matplotlib - for plotting graphs
- SVM –python library used for multiclass classification
- TQDM -shows the progress of any iteration of epochs 
- numpy - for array computation
- Gensims- for topic modelling, document indexing and similarity retrieval for large corpora.
- nltk (Natural Language Toolkit) –suite of libraries and programs for symbolic and statistical natural language processing in python

## Flow Diagram
![image](https://user-images.githubusercontent.com/43497595/120112057-eb7a1f80-c191-11eb-9bf3-4ab33d712bd2.png)

## Analysis
![image](https://user-images.githubusercontent.com/43497595/120112132-35fb9c00-c192-11eb-9df7-890698d359d8.png)

## Refrences
- Research Paper:  Fake news detection within online social media using supervised artificial intelligence algorithms Feyza Altunbey Ozbay, Bilal Alatas , Department of Software - Engineering, Faculty of Engineering, Firat University, 23100, Elazig, Turkey
- Research Paper: Advanced Machine Learning techniques for fake news (onlinedisinformation) detection: A systematic mapping study
- Research Paper: Detecting Fake News in Social Media Networks. Monther Aldwairi, Ali Alwahedi
- Research Paper: A Survey of Fake News
- Research Paper: Fake news detection in social media Kelly Stahl
- Research Paper: Fake News Detection System Using Logistic Regression Technique In Machine Learning
- Research Paper:  FAKE NEWS DETECTION USING LOGISTIC REGRESSION Fathima Nada
- Research Paper:  Fake News Detection Using Logistic Regression, Sentiment Analysis and Web Scraping
- Research Paper:  Fake News Detection on Social Media: A Data Mining Perspective
- Research Paper: Linguistic feature based learning model for fake news detection and classification
- https://www.kaggle.com/mrisdal/fake-news/data
- https://open-platform.theguardian.com/
- https://developer.nytimes.com/
- https://www.youtube.com/channel/UCObs0kLIrDjX2LLSybqNaEA
- https://www.youtube.com/c/yobots/videos
- https://towardsdatascience.com/



