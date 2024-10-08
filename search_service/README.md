# summary

1. Unload data by running `docker-compose up` at `../docker-workspace`
1. Unload data and model by running `docker-compose up` at `search_service/data_dir`
1. Deploy containers by running `docker-compose up` at `search_service`

## Deployment Instruction

First, get docker-compose version >= v1.28.0. Easiest way is to do `pip install docker-compose` and check version `docker-compose --version`.

Second, git clone this repo, and download data/model. *In the same folder as this md file*, run:
```
cd data_dir
docker-compose up
cd ..
```

Third, run the search containers. *In the same folder as this md file*, run:
```
export DATA_DIR="$(pwd)/data_dir/"
docker-compose up
```
## Notice
The "stylebot_wild_seg" containers has a directory(`/stylebot_wild_seg/`) where the ML model works. In order to get the latest version, run
```
git clone https://github.com/AIML-K/stylebot_wild_seg.git
```