GRANT CONNECT,RESOURCE,UNLIMITED TABLESPACE TO STUDYY IDENTIFIED BY TIGER;
ALTER USER STUDYY DEFAULT TABLESPACE USERS;
ALTER USER STUDYY TEMPORARY TABLESPACE TEMP;
CONNECT STUDYY/TIGER
DROP TABLE DEPT;

.header on
.mode column

CREATE TABLE mountains (
  name TEXT,
  height_meters INTEGER,
  first_ascent DATE,
  first_human TEXT
);

INSERT INTO mountains VALUES
  ('Mount Everest', 8848, '1953-05-29','KimAmugae'),
  ('Kilimanjaro', 5895, '1889-10-06','Sunyoung'),
  ('Denali', 6190, '1913-06-07', 'Moon'),
  ('Chimborazo', 6263, '1880-01-04','Sunyoung'),
  ('K2', 8611, '1954-07-31', NULL),
  ('Piz Palü', 3900, '1835-08-12', null),
  ('Cho Oyu', 8848, '1954-10-19', 'juyoung');

.print 'average mountain height'
SELECT avg(height_meters) AS avg_height
FROM mountains;

.print
.print 'number of ascents per century'
SELECT
  strftime(
    '%Y',
    date(first_ascent)
  ) / 100 + 1 AS century,
  count(*) AS ascents
FROM mountains
GROUP BY century;


--문제1. 정복자의 데이터를 갖고있는 산의 이름만 출력하라

--문제2. 첫 등반날짜가 1800년대인 산의 이름과 고도를 출력하라

--문제3. 정복자 기준으로 오름차순으로 정리하여 모두 출력하고 산의 이름이 같다면 산에 대해서도 오름차순이 될 수 있도록 하라.

--문제4. 정복자가 없는 산의 이름과 고도를 고도가 높은 순서대로 출력하라

--문제5. 산이름 중간에 공백이 있는 산의 이름과 정복자를 출력하라



