drop table book cascade constraint;
drop sequence seq_book_no;
create sequence seq_book_no;

create table book(
    bk_no number(6) primary key,
    title varchar2(20) not null,
    author varchar2(20) not null,
    price number(6,2) not null
);

insert into book values(seq_book_no.nextval, 'python', 'bansome', 1000);

commit;

select * from book;
