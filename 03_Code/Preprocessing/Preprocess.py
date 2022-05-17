import re
import arabicstopwords.arabicstopwords as stp
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


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
