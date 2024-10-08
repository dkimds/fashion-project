# How to Unload data
## Step
1. 데이터를 받게해주는 도커 이미지를 aimlk-dockerhub organization에서 pull 해옵니다.

1. 터미널에서 docker-workspace로 이동합니다.

1. 이미지 pull이 끝나면, `docker-compose up` 커맨드를 입력합니다.
                그 결과 1) 훈련데이터가 담긴 category_210907 디렉토리
                        2) 데이터에대한 설명이 있는 description_data.md
                        3) 라벨에 대한 세부 정보가 담긴 엑셀 파일이 데이터가 생성됩니다.
## Warning
또, preprocess.py를 실행함에 있어 user가 root으로 지정되어 오류가 발생할 것입니다.
이는 stylebot_wild_seg/docker-workspace 에서 아래 커맨드를 입력하면 되겠습니다.
```
sudo chown $(id -u):$(id -g) -R . 
```
