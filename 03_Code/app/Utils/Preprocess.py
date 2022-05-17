import re
import numpy as np
import string
import requests
from bs4 import BeautifulSoup as bs
from bs4.element import Comment
import arabicstopwords.arabicstopwords as stp
from os.path import exists
import codecs
from collections import Counter
from nltk.corpus import stopwords
import pandas as pd
stop_words = set(stopwords.words('english'))


def is_file_exist(file_path):
    return exists(file_path)


def read_html_page(html_path):
    return bs(html_path, 'html.parser')


def read_file(file_path):
    try:
        file = codecs.open(file_path, 'r', encoding="utf-8").read()
    except UnicodeDecodeError:
        file = codecs.open(file_path, 'r', encoding="windows-1256").read()
    return file


# def tag_visible(element):
#     if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]', 'comment']:
#         return False
#     if isinstance(element, Comment):
#         return False
#     return True


# def text_from_html(soup):
#     texts = soup.findAll(text=True)
#     visible_texts = filter(tag_visible, texts)
#     return " ".join(t.strip() for t in visible_texts)

def text_from_html(soup):
    return soup.get_text()


def check_response_status(url):
    response = None
    status = 200
    try:
        response = requests.get(url)
        response.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        #print("Down")
        status = response.status_code
    except requests.exceptions.HTTPError:
        #print("4xx, 5xx")
        status = response.status_code
    except requests.exceptions.RequestException :
        #print("Error")
        status = response.status_code
    except:
        status = 0
    else:
        #print("Request Status Is good")
        status = response.status_code
    finally:
        return status, response


def check_sample_online_request(url):
    response_status, response = check_response_status(url)
    soup = None
    if response is not None:
        soup = bs(response.text, 'html.parser')
    return response_status, soup


def remove_emoji(text):
    regex_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  
                                u"\U0001F300-\U0001F5FF"  
                                u"\U0001F680-\U0001F6FF"  
                                u"\U0001F1E0-\U0001F1FF"  
                                u"\U00002500-\U00002BEF"  
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642" 
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  
                                u"\u3030"
                               "]+", flags=re.UNICODE)

    return regex_pattern.sub(r'', text)


def remove_email(text):
    return re.sub('([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', '', text)


def remove_repeated_char(text):
    return re.sub(r'(.)\1\1{1,}', r'\1\1', text)


def remove_account_tag(text):
    return re.sub(r'@[\w]+', '', text)


def remove_hashtag(text):
    return re.sub(r'#[\w]+', '', text)


def remove_more_spaces(text):
    return re.sub('\s+\t\n\r', ' ', text)


def remove_stop_words(text):
    text_list = []
    for w in text.split():
        if (not stp.is_stop(w)) and (w not in stop_words):
            text_list.append(w)
    return " ".join(text_list)


def clean_text(text):
    text = text.lower()
    text = remove_emoji(text)
    text = remove_email(text)
    text = remove_account_tag(text)
    text = remove_hashtag(text)
    text = remove_stop_words(text)

    text = re.sub(r'http\S+', '', text)

    tags_comp = re.compile('<.*?>')
    text = re.sub(tags_comp, '', text)

    text = re.sub(r'[^\w\s]', ' ', text)

    text = re.sub(r'\w*\d\w*', ' ', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = remove_more_spaces(text)

    text = " ".join([x for x in text.split() if len(x) > 3])

    text = " ".join([x for x in text.split() if not x.isdigit()])

    return text


def get_page_tage_text(soup_src, tag_name):
    tag_content = soup_src.find_all(tag_name)
    for i in range(len(tag_content)):
        tag_content[i] = clean_text(tag_content[i].get_text())
    return tag_content


def get_unique_words(text):
    return " ".join(list(dict.fromkeys(text.split())))


def get_tag_unique_content(soup, tag):
    text = get_page_tage_text(soup, tag)
    text = " ".join(text)
    text = clean_text(text)
    text = get_unique_words(text)
    return text


def get_n_top_freq_words(text, n):
    split_it = text.split()
    Counter_val = Counter(split_it)
    most_occur = Counter_val.most_common(n)
    return [x[0] for x in most_occur]


def check_domain_url(soup, domain_name):
    hrefs = [f.get('href') for f in soup.find_all('link')]
    for link in hrefs:
        if link is not None and domain_name in link:
            return True

    hrefs = [f.get('href') for f in soup.find_all('a')]
    for link in hrefs:
        if link is not None and domain_name in link:
            return True

    forms = [f.get('action') for f in soup.find_all('form')]
    for link in forms:
        if link is not None and domain_name in link:
            return True
    return False


def get_key_occurance_text(text, key):
    if key is None or key == "":
        return 0
    return text.count(key)


def get_n_related_scores(text, keys, delim="-|_|,", n=3):
    key_list = re.split(delim, keys)
    res = np.zeros((n,))
    for i, key in enumerate(key_list):
        res[i] = get_key_occurance_text(text, key.strip())
    return res


def get_page_features(soup, row, n_content_names=3):
    domain_name_status = int(check_domain_url(soup, row["link_domain_name"]))
    text = text_from_html(soup)
    content_name_freq = get_key_occurance_text(text, row["content_name"])
    trans_content_name_freq = get_key_occurance_text(text, row["trans_content_name"])
    alt_content_names_freq_list = get_n_related_scores(text, row["alt_content_names"], n=n_content_names)
    campaign_name_freq = get_key_occurance_text(text, row["campaign_name"])

    return domain_name_status, content_name_freq, trans_content_name_freq, alt_content_names_freq_list, campaign_name_freq


def generate_clean_df(df):
    top_n_word = 5
    df["response_status"] = 200
    df["domain_name_status"] = 0
    df["content_name_freq"] = 0
    df["trans_content_name_freq"] = 0
    n_content_names = 5
    for i in range(n_content_names):
        df["alt_content_names_{}".format(i)] = None
    df["campaign_name_freq"] = None
    df["title"] = None
    df["h1"] = None
    df["h2"] = None
    df["h3"] = None
    df["h4"] = None
    df["h5"] = None
    df["h6"] = None
    df["content"] = None
    df["top_{}_word".format(top_n_word)] = None

    for index, row in df.iterrows():
        # print(index)
        # response_status, soup = check_sample_online_request(row["link_url"])
        # if soup is not None:
        #    df["response_status"] = response_status
        # else:
        #    if(not is_file_exist(row["page_source_path"])):
        #        continue
        #    soup = read_html_page(read_file(row["page_source_path"]))

        # print(index)
        if is_file_exist(row["page_source_path"]):
            check_online = False
            soup = read_html_page(read_file(row["page_source_path"]))

            if soup is None:
                check_online = True

            elif soup.title is None or soup.title.string is None or "I am not a bot. Open Website" in soup.title.string:
                check_online = True

            elif len(text_from_html(soup)) == 0:
                check_online = True

            if check_online:
                response_status, soup = check_sample_online_request(row["link_url"])
                df.at[index, "response_status"] = response_status
                if soup is None:
                    df.at[index, "response_status"] = 0
                    continue
        else:
            response_status, soup = check_sample_online_request(row["link_url"])
            df.at[index, "response_status"] = response_status
            if soup is None:
                df.at[index, "response_status"] = 0
                continue

        domain_name_status, content_name_freq, trans_content_name_freq, alt_content_names_freq_list, campaign_name_freq = get_page_features(soup, row, n_content_names)
        df.at[index, "domain_name_status"] = domain_name_status
        df.at[index, "content_name_freq"] = content_name_freq
        df.at[index, "trans_content_name_freq"] = trans_content_name_freq
        for i in range(n_content_names):
            df.at[index, "alt_content_names_{}".format(i)] = alt_content_names_freq_list[i]
        df.at[index, "campaign_name_freq"] = campaign_name_freq

        df.at[index, "title"] = " ".join(get_page_tage_text(soup, "title"))
        df.at[index, "h1"] = get_tag_unique_content(soup, "h1")
        df.at[index, "h2"] = get_tag_unique_content(soup, "h2")
        df.at[index, "h3"] = get_tag_unique_content(soup, "h3")
        df.at[index, "h4"] = get_tag_unique_content(soup, "h4")
        df.at[index, "h5"] = get_tag_unique_content(soup, "h5")
        df.at[index, "h6"] = get_tag_unique_content(soup, "h6")

        text = text_from_html(soup)
        text = clean_text(text)
        df.at[index, "content"] = text

        df.at[index, "top_5_word"] = " ".join(get_n_top_freq_words(text, top_n_word))
    return df


def json_2_df(json):
    return pd.DataFrame.from_records(json)
