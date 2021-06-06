--12.sequence.sql
/*
1. 시퀀스 
	: 순차적인 순서 번호를 자동으로 반영할수 있는 매우 유용한 기술
	: 기본은 1씩 자동 증가
		- 증가치, 최대값 추가 설정도 가능
		- 권장 : 하나의 시퀀스를 다수의 table에서 사용 비추

2. 대표적인 활용 영역
	- 게시물 글번호에 주로 사용
	- 고객 등록시에 1씩 자동 증가해서 고객 관리
3. seq 기능
	- 일정 숫자 만큼 자동 생성되는 특성
	- 속성
		nextval : insert문에서 사용 / 1씩 자동 증가해서 insert
		currval : 현시점에서 seq의 값을 
*/	

-- 어떤 table의 어떤 column에 sequence 적용할 것인지 설계를 명확한 상태에서 진행
--   
--1. sequence 생성 명령어
create sequence seq_test_no1;

drop table test;
create table test(
	no1 number(3)
);

--2. seq~를 활용한 insert
insert into test values(seq_test_no1.nextval);
insert into test values(seq_test_no1.nextval);
insert into test values(seq_test_no1.nextval);
insert into test values(seq_test_no1.nextval);
insert into test values(seq_test_no1.nextval);

-- 해당 시퀀스 현재값 검색
select seq_test_no1.currval from dual;

--3. 다수의 table에서 하나의 seq를 공동 사용시: sequence는 공유가 된다
-- 따라서 table당 고유한 sequence 값 즉 1씩 증가치를 보장해야 할 경우엔 공유 금지 

create table test3(
	no1 number(3)
);

insert into test3 values(seq_test_no1.nextval);
insert into test values(seq_test_no1.nextval);
insert into test3 values(seq_test_no1.nextval);

select * from test;
select * from test3;


--4. 시작 index 지정 및 증가치도 지정하는 seq 생성 명령어
-- 1씩 자동증가치가 아닌 10으로 시작해서 2씩 자동 증가 단 최대 20까지만 시퀀스 사용 의미 
-- 속성이 생략될 경우 1씩 자동 증가가 default
drop sequence seq_test_no1;
create sequence seq_test_no1
start with 10
increment by 2
maxvalue 20;


--5. 현 sequence의 데이터값 검색하기
select seq_test.currval from dual;


--6. seq 삭제 명령어
drop sequence seq_test;
