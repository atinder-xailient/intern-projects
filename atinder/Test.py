def find_direct_url():
    import requests
    from bs4 import BeautifulSoup as bs
    
    url="https://www.youtube.com/watch?v=cWzkAHB1kT8&feature=emb_logo"
    source = requests.get(url).content
    soup = bs(source, from_encoding='utf-8', features="lxml")
    tags = soup.findAll("script")
    
    imp_tags = []
    for tag in tags:
        if tag.string is not None and "m3u8" in tag.string:
            imp_tags.append(tag)
    
    imp_strings = []
    for tag in imp_tags:
        split_tag = tag.string.split('\"')
        for string in split_tag:
            if "m3u8" in string:
                imp_strings.append(string)
    
    url = imp_strings[0]
    url = url.replace('\\','')
    return url
