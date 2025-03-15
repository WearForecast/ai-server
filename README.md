# WearForecast
<img width="1123" alt="image" src="https://github.com/user-attachments/assets/2ca75475-cca5-4da0-8911-4d2f713803bb" />

# WearForecast AI Server
이 저장소는 **FastAPI**를 사용하여 날씨와 성별을 기반으로 의류를 추천하는 API를 제공합니다. **CLIP 모델**과 **Gemini**를 활용하여 텍스트 기반 추천을 수행하고, **Supabase** 데이터베이스에서 적절한 이미지를 검색하여 사용자에게 추천합니다.  

## 시작 가이드
### Requirements
- Python 3.8 이상
- **Conda** (가상 환경 관리) 
- API 키 및 데이터베이스 정보 (`.env` 파일 필요)

### Installation
1. **저장소 클론**  
```bash
$ git clone https://github.com/WearForecast/ai-server.git
```
2. **Conda 가상 환경 생성**
```bash
$ conda create -n fashion-ai python=3.10
$ conda activate fashion-ai
```
3. **필수 패키지 설치**
```
$ pip install -r requirements.txt
```
4. **환경 변수 설정**
```
API_KEY=YOUR_GEMINI_API_KEY
SUPABASE_URL=YOUR_SUPABASE_URL
SUPABASE_KEY=YOUR_SUPABASE_KEY
```
5. **서버 실행**
```
$ uvicorn main:app --reload
```
6. **API 문서 확인**
서버가 실행되면 http://127.0.0.1:8000/docs 에서 API 문서를 확인할 수 있습니다.

## 주요 기능  

### 날씨와 성별 기반 의류 추천  
- 날씨(예: `"맑음, 15°C"`), 성별( `"남성"` 또는 `"여성"` )을 입력하면 AI가 적절한 의상을 추천합니다.  

### 자연스러운 한국어 번역  
- AI가 생성한 추천 문장을 한국어로 번역하여 제공하며, 색상 및 성별 정보를 제외하여 보다 자연스러운 표현을 생성합니다.  

### 이미지 검색 및 매칭  
- CLIP 모델을 이용해 추천 문장을 벡터로 변환 후, Supabase에서 가장 적절한 의류 이미지를 검색하여 제공합니다.

## 아키텍쳐
### 프로젝트 구조
```plaintext
ai-server
├─ .dockerignore
├─ Dockerfile
├─ README.md
├─ fine-tune-fashionclip
│  ├─ fine-tune-fashionclip.py
│  ├─ generate-labels.py
│  └─ kfashion-dataset.csv
├─ image_scraper
│  └─ crawler.py
├─ main.py                                 # FastAPI 서버 실행 파일
├─ model
│  ├─ generate-clip-embeddings.py
│  ├─ image_embeddings.csv
│  └─ model.py                             # 의류 추천 모델 (CLIP + Gemini + Supabase)
└─ requirements.txt
```
