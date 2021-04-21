
import requests, pymongo, json, time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup




client = pymongo.MongoClient()



def init_mongo_client():
    
    client = pymongo.MongoClient()
    db = client.nyt
    coll = db.NYT_Articles
    return db.articles


def call_api(url, payload, p=0):
    payload['page'] = p
    return requests.get(url, params=payload)


def get_response(r):
    raw = json.loads(r.text)
    return raw['response']['meta'], raw['response']['docs']


def get_soup(url):
    agent  = 'DataWrangling/1.1 (http://zipfianacademy.com; '
    agent += 'class@zipfianacademy.com)'
    headers = {'user_agent': agent}

    try:
        r = requests.get(url, headers=headers)
    except:
        return None

    if r.status_code != 200: return None
    return BeautifulSoup(r.text.encode('utf-8'),"lxml")


def get_body_text(docs):

    result = []
    for d in docs:
        doc = d.copy()
        if not doc['web_url']:
            continue

        soup = get_soup(doc['web_url'])
        if not soup:
            continue


        body = soup.find_all('p', class_= "css-xhhu0i")
        if not body:
            continue


        doc['body'] = '\n'.join([x.get_text() for x in body])

        print (doc['web_url'])
        result.append(doc)

    return result


def remove_previously_scraped(coll, docs):
    
    new_docs = []
    for doc in docs:
        
        cursor = articles.find({'_id': doc['_id']}).limit(1)
        if not cursor.count() > 0:
            new_docs.append(doc)

    if new_docs == []:
        return None

    return new_docs


def get_end_date(dt):

    yr   = str(dt.year)
    mon = '0' * (2 - len(str(dt.month))) + str(dt.month)
    day = '0' * (2 - len(str(dt.day))) + str(dt.day)
    return yr + mon + day


def scrape_articles(coll, last_date):
    page = 0
    while page <= 199:
        print ('Page:', page)

        payload  = {'sort': 'newest',
                    'end_date': get_end_date(last_date),
                    'api-key': API_KEY
                   }
        
        r = call_api(NYT_URL, payload, page)
        page += 1
        if r.status_code != 200:
            page = 0
            last_date += relativedelta(days=-1)
            print ('End Date:', get_end_date(last_date))
            print (r.status_code )
            time.sleep(2)
            continue
            

        meta, docs = get_response(r)

        new_docs = remove_previously_scraped(coll, docs)
        if not new_docs: continue

        docs_with_body = get_body_text(new_docs)

        for doc in docs_with_body:
            try:
                coll.insert_one(doc)
            except:
                continue



NYT_URL = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
api_key_path = "E:/Projects/ML_Project/FakeNewsDetection/FetchingData/NYT_API_Key.txt"


with open('NYT_API_Key.txt', 'r') as handle:
    API_KEY = handle.read()


articles = init_mongo_client()
last_date = datetime.now() + relativedelta(days=-2)
scrape_articles(articles, last_date)
