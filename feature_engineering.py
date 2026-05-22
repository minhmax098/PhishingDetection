import re
import urllib.parse
import math
from collections import Counter
import pandas as pd
import numpy as np

# List of suspicious keywords commonly found in phishing URLs
SUSPICIOUS_KEYWORDS = ['login', 'verify', 'update', 'secure', 'account', 'password', 'banking', 'confirm']

def get_url_length(url):
    return len(str(url))

def get_num_dots(url):
    return str(url).count('.')

def get_num_hyphens(url):
    return str(url).count('-')

def get_num_slashes(url):
    return str(url).count('/')

def get_num_digits(url):
    return sum(c.isdigit() for c in str(url))

def get_num_special_chars(url):
    special_chars = ['@', '%', '&', '=', '?', '_']
    return sum(str(url).count(c) for c in special_chars)

def has_at_symbol(url):
    return 1 if '@' in str(url) else 0

def has_ip_address(url):
    # Simple regex to check for IPv4 presence
    if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', str(url)):
        return 1
    return 0

def get_num_suspicious_keywords(url):
    url_lower = str(url).lower()
    return sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)

def shannon_entropy(string):
    """Calculates the Shannon entropy of a string."""
    if not string:
        return 0.0
    counts = Counter(string)
    frequencies = ((i / len(string)) for i in counts.values())
    return - sum(f * math.log(f, 2) for f in frequencies)

def extract_features_from_dataframe(df, url_col='url'):
    """
    Extracts features for an entire dataframe.
    """
    df = df.copy()

    # URL Features
    df['url_length'] = df[url_col].apply(get_url_length)
    df['n_dots'] = df[url_col].apply(get_num_dots)
    df['n_hyphens'] = df[url_col].apply(get_num_hyphens)
    df['n_slashes'] = df[url_col].apply(get_num_slashes)
    df['n_digits'] = df[url_col].apply(get_num_digits)
    df['n_special_chars'] = df[url_col].apply(get_num_special_chars)
    df['has_at_symbol'] = df[url_col].apply(has_at_symbol)
    df['has_ip_address'] = df[url_col].apply(has_ip_address)
    df['n_suspicious_keywords'] = df[url_col].apply(get_num_suspicious_keywords)

    # Domain Features
    def parse_domain(url):
        url_str = str(url)
        parsed = urllib.parse.urlparse(url_str if "://" in url_str else "http://" + url_str)
        return parsed.netloc

    domains = df[url_col].apply(parse_domain)

    df['domain_length'] = domains.apply(len)
    df['n_subdomains'] = domains.apply(lambda d: max(0, d.count('.') - 1) if not has_ip_address(d) else 0)
    df['domain_entropy'] = domains.apply(shannon_entropy)

    # Placeholders for Host-based Features (mocked to 0 for now)
    df['domain_age_days'] = np.random.randint(1, 3650, size=len(df)) # Mock age
    df['has_ssl'] = np.random.choice([0, 1], size=len(df))          # Mock SSL
    df['has_dns'] = np.random.choice([0, 1], size=len(df))          # Mock DNS

    return df

if __name__ == "__main__":
    df = pd.DataFrame({'url': ['http://secure-login.paypal.com.xyz', 'http://example.com']})
    features_df = extract_features_from_dataframe(df)
    print(features_df.head())
