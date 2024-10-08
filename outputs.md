# outputs
산출물 목록
- 코드
- 도커 이미지
- 테이블

## 코드 
압축파일로 제공
### 데이터 언로더
- stylebot_sb1_data_unloader.zip: 데이터 패키징
    - docker-compose.yml: 데이터 언로드 실행

### 딥러닝 모델
- stylebot_wild_seg.zip: 코디네이션 인식
    - 피클: 코디테이션 데이터
        - cls_id: 딥러닝 모델의 씽 클래스 아이디
    - 이미지: 피클 시각화

## 도커 이미지
docker-compose.yml로 도커 허브에서 이미지 다운로드
### 데이터
- 이미지 네임: `stylebot_sb1_data_unloader`
- pull command
```
docker pull aimlk/stylebot_sb1_data_unloader:0.0.6
```
### 콘테이너
- 이미지 네임: `stylebot_wild_seg`
- pull command

```
docker pull aimlk/stylebot_wild_seg:v1.0.1-cuda111
```

## 테이블
stylebot_sb1_data_unloader 이미지에 엑셀파일로 제공

### 스타일봇 카테고리 기준_ 착장 데이터 폴더링 구조.xlsx
- sheet1: 세부 카테고리와 딥패션 카테고리를 매핑
    - 세부 카테고리: 스타일봇 카테고리 depth2에서 연구용으로 분류한 41개 복종 
    - 딥패션 카테고리: 해외 논문에서 분류한 13개 복종
- sheet2: stylebot_wild_seg의 씽 클래스와 스타일봇 카테고리를 매핑
    - 씽 클래스: 딥러닝 모델이 인식하는 세부 카테고리 
    - 스타일봇 카테고리: depth2