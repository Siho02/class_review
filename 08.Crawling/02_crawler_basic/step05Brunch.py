import re
from urllib.request import urlopen
from html import unescape

def main():
    html = crawling('https://brunch.co.kr/')
    #print(html)
    brunches = scraping(html)
    #print(brunches)

def crawling(url):
    f = urlopen(url)

    encoding = f.info().get_content_charset(failobj="utf-8")
    html = f.read().decode(encoding)

    return html 

def scraping(html):
    '''
        <td class="left"><a href="/store/books/look.php?p_code=B4300598719">리눅스 입문자를 위한 명령어 사전</a></td>
    '''
    # for data in re.findall(r'<td class="left"><a.*?/td>' , html):


    '''
        <span class="keyword_item_txt">사진·촬영</span>
        for data in re.findall(r'<span class=".*?/span>', html):
            print(data)
            #d1 = re.search(r'class="keyword_item_txt"(.*)', data[0])
            #print(d1)
            #d2 = re.sub('"', '', d1)  
    '''



if __name__ == "__main__":
    main()