from book_info import Book 
import cx_Oracle
#클라이언트 화면을 통해서 책 정보를 등록
#데이터 획득 BOOK객체 생성
#오라클 DB에 저장

def book_insert(book):
    # oracle db에 insert
    conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
    cur = conn.cursor() 
    print(cur)
    #insert into book values(seq_book_no.nextval, 'python', 'bansome', 1000);
    cur.execute("insert into book values(seq_book_no.nextval, :title, :author, :price)", title=book.getTitle(), author=book.getAuthor(), price=book.getPrice())
    conn.commit()

    cur.close()
    conn.close()

def book_update(book):
    # oracle db에 insert
    conn = cx_Oracle.connect(user="SCOTT", password="TIGER", dsn="xe")
    cur = conn.cursor() 
    cur.execute("update book set author = :author, price = :price where title = :title", \
                author = book.getAuthor(),title=book.getTitle(), price=book.getPrice())
    conn.commit()

    cur.close()
    conn.close()

if __name__ == '__main__':
    #book_insert(Book('python2', '로드반', 2000))
    book_update(Book('python2', 'load lee', 3500))