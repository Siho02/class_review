from datetime import datetime
from elasticsearch import Elasticsearch
# es를 사용가능한 python객체 생성, 접속, doc 생성, 
es = Elasticsearch()

def put():
    '''
        doc라는 변수에 3개의 field 선언해서 값 설정
        python 자체적으로 dict 타입
        es 관점에선 field와 value
        datetime.now() : 현 날짜 시간 표현
    '''
    doc = {
        'author': 'kimchy',
        'text': 'Elasticsearch: cool. bonsai cool.',
        'timestamp': datetime.now(),
    }

    '''
    es.index(index="test-index", id=1, body=doc)
    es : Elasticsearch 객체
    index() : ES 보유한 index 생성함수
    * 용어 정리 : index 생성은 RDBMS 관점에서 table 생성 + 데이터 저장
    inddex = :index명
    test-index : index로 사용할 이름을 설정
    id : pk와 같이 고유한 id값 설정 의미
    =1 : id값으로 1 의미
    body : index에 설정한 table과 같은 index구조에 id값과 저장될 데이터 설정하는 속성
    =doc : dict구조의 변수값으로 idnex에 새로운 데이터 생성 / 이미 존재할 경우 update
    '''

    res = es.index(index="test-index", id=1, body=doc)
    print('--',res['result'])



def get():
    res = es.get(index="test-index", id=1)
    print(res['_source'])

def match_all():
    res = es.search(index="test-index", body={"query": {"match_all": {}}})
    print("Got %d Hits:" % res['hits']['total']['value'])

    for hit in res['hits']['hits']:
        print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])

if __name__ == '__main__':
    get()
    #match_all()