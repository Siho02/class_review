from elasticsearch import Elasticsearch
es = Elasticsearch()

def get():
    res = es.get(index="bank", id=1)
    print(res['_source'])

'''
{
  "query": {
    "match": {
      "bank": "국민은행"
    }
  },
  "size": 10, 
  "aggs": {
    "b_1": {
      "terms": {
        "field": "customers"
      }
    }
  }
}
'''

def cus_branch():
    res = es.search(index="bank", body={"query": { "match" : {"bank" : "국민은행"}}, "size":10, "aggs":{"b_1":{"terms":{"field":"customers"}}}})
    print(res)
    print(res['hits']['total']['value'])
    print(res['hits']['total']['relation'])

def bank_get():
    res = es.search(index="bank", body={"query": {"match": { "bank": "국민은행"}}})
    print(res)

    #['hits'] >> ['total'] >> ['value'] : 조상 >> 부모 >> 자식
    print("집계 개수 %d" % res['hits']['total']['value'])
    # print(res['aggregations'])


def bank_get2():
    res = es.search(index="bank", body={"query": { "match": { "bank": "국민은행" }},"size": 0,  "aggs": { "b_1": {"terms": { "field": "customers"}}}})
    #print(res)
    print(res['aggregations']['b_1']['buckets'])

    #for bucket in ['aggregations']['b_1']['buckets']:
    #    bucket['key']
    for bucket in res['aggregations']['b_1']['buckets']:
        print(bucket['key'])

def bank_unique():
    res = es.search(index="bank", body={"size": 0, "aggs": {"b_2": {"cardinality": {"field": "branch.keyword"}}}})
    print(res)
    print("은행 종류 : ", res['aggregations']['b_2']['value'])

def kb_bank():
    res = es.search(index='bank', body={"query": {"match": {"bank": "국민은행"}},"size": 0,"aggs": {"b_3": {"stats": {"field": "customers"}}}})
    print(res['aggregations']['b_3'])

if __name__ == '__main__':
    #bank_get()
    #bank_get2()
    #bank_unique()
    kb_bank()