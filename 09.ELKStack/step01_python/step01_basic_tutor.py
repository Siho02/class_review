from datetime import datetime
from elasticsearch import Elasticsearch

# ES를 사용 가능한 python 객체 생성, 접속, doc생성, 검색등이 다 가능한 기능을 보유한 객체
# 이 소스가 실행중인 시스템(해당 ip)에서 실행되는 ES 에 자동 접속
es = Elasticsearch()

def put():
    '''
        doc 라는 변수에 3개의 field 선언해서 값 설정
        python 자체적으론 dict 타입
        es관점에선 field와 value
        datetime.now() : 현 날짜시간 표현
    '''
    doc = {
        'author': 'kimchy',
        'text': 'Elasticsearch: cool. bonsai cool.',
        'timestamp': datetime.now(),
    }

    '''
        es.index(index="test-index", id=1, body=doc)
        es : Elasticsearch 객체
        index() : ES보유한 index 생성 함수
        * 용어 정리 : index 생성은 RDBMS 관점에서 table 생성 + 데이터 저장 
        index = : index 명
        test-index : index로 사용할 이름을 설정
        id : pk와 같이 고유한 id값 설정 의미
        =1 : id값으로 1 의미
        body : index에 설정한 table과 같은 index구조에 id값과 저장될 데이터 설정하는 속성
        =doc : dict 구조의 변수값으로 index에 새로운 데이터 생성 . 이미 존재할 경우 update
    '''
    res = es.index(index="test-index", id=1, body=doc)
    print(res['result'])   # create  or update

def get():
    # GET test-index/_doc/1
    res = es.get(index="test-index", id=1)
    print(res['_source'])

def match_all():
    '''
    GET test-index/_search
    {
        "query": {
            "match_all": {}
        }
    }
    '''
    res = es.search(index="test-index", body={"query": {"match_all": {}}})
    
    # %d 가변적인 데이터 단 d는 디지털 약자 즉 숫자 의미 
    print(res)
    print('-------')
    print("Got %f Hits:" % res['hits']['total']['value'])

    for hit in res['hits']['hits']:

        ''' 키바나 결과
        "_source" : {
            "author" : "kimchy",
            "text" : "Elasticsearch: cool. bonsai cool.",
            "timestamp" : "2021-07-13"
            }
        '''
        # 콘솔창에 출력된 결과 
        #2021-07-13 kimchy: Elasticsearch: cool. bonsai cool.
        # %(키)s %(키2)s     %  dict 구조의 데이터의 해당 key와 일치되는 데이터를 문자열 포멧에 자동 적용 
        print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])


if __name__== "__main__":
    #put()
    #get()
    match_all()