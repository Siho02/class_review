import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform


def visualization():
    if platform.system() == 'Windows':
        path = 'c:/Windows/Fonts/malgun.ttf'
        font_name = font_manager.FontProperties(fname = path).get_name()
        rc('font', family = font_name)
    elif platform.system() == 'Darwin':
        rc('font', family = 'AppleGothic')
    else:
        print('Check your OS system')

    #rank_crawl.py를 통해 저장한 excel 파일을 읽어옴
    df = pd.read_excel('./files/youtube_rank.xlsx')

    #데이터 프레임 전처리
    df['replaced_subscriber'] = df['subscriber'].str.replace('만', '0000')
    df['replaced_view(단위:만)'] = df['view'].str.replace('억', '')
    df['replaced_view(단위:만)'] = df['replaced_view(단위:만)'].str.replace('만', '')

    #구독자 수에 따른 유튜브 버튼을 나누는 함수 지정
    sub_num = ''
    def range_sep(num):
        if num >= 10000000:
            sub_num = 'diamond button'
        elif num >= 500000:
            sub_num = 'gold button'
        elif num >= 100000:
            sub_num = 'silver button'
        else :
            sub_num = 'under_silver'
        return sub_num

    #구독자 수의 데이터 타입을 str >> int로 변환하여 첫번째 pivot을 만듦
    df['replaced_subscriber'] = df['replaced_subscriber'].astype('int')
    df['replaced_view(단위:만)'] = df['replaced_view(단위:만)'].astype('int')
    df['button_sub'] = df['replaced_subscriber'].apply(range_sep) 
    pivot_df = df.pivot_table(index = 'button_sub', values = 'replaced_subscriber', aggfunc = ['sum','count'])
    pivot_df.columns = ['subscriber_sum', 'category_count']
    pivot_df = pivot_df.reset_index()

    #조회수(view)를 이용한 pivot table 만듦
    pivot_df2 = df.pivot_table(index = 'button_sub', values = 'replaced_view(단위:만)', aggfunc=['sum','mean'])
    pivot_df2.columns = ['view_sum', 'view_mean']
    pivot_df2 = pivot_df2.reset_index()

    print(pivot_df)
    print(pivot_df2)

    #만든 pivot table들을 차트로 만들어 저장
    plt.figure(figsize = (10,10))
    plt.pie(pivot_df['subscriber_sum'], labels=pivot_df['button_sub'], autopct='%1.1f%%')
    plt.savefig('C:/202105_lab/08.Crawling/06_webapp_step02/static/image/save1.png')

    plt.figure(figsize = (10,10))
    plt.pie(pivot_df['category_count'], labels=pivot_df['button_sub'], autopct='%1.1f%%')
    plt.savefig('C:/202105_lab/08.Crawling/06_webapp_step02/static/image/save2.png')

    plt.figure(figsize = (10,10))
    plt.pie(pivot_df2['view_mean'], labels=pivot_df['button_sub'], autopct='%1.1f%%')
    plt.savefig('C:/202105_lab/08.Crawling/06_webapp_step02/static/image/save3.png')

    plt.figure(figsize = (10,10))
    plt.pie(pivot_df2['view_sum'], labels=pivot_df['button_sub'], autopct='%1.1f%%')
    plt.savefig('C:/202105_lab/08.Crawling/06_webapp_step02/static/image/save4.png')

    return True
