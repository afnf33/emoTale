# emo:)Tale
emo:)Tale 이란?   
emotion(감정) + tale(이야기). 이야기에 담긴 감정을 찾아보자!

양재 AI 허브에서 진행한 2020 여름 AI College의 팀프로젝트

프로젝트 제목: 트위터 기반 한국어 자연어 다중 감성 분석

팀원: 김영운, 김태희, 박건우, 변자민, 신승현, 윤보현, 이성준, 장승현

<br>

# 1. 프로젝트 개요

인공지능 자연어 처리에서 감성 분석은 작게는 제품 사용자의 후기 분석을 통한 시장 반응 조사에서부터 크게는 소셜미디어 분석을 통한 대선 결과 예측까지 사회의 거시적 트렌드를 분석해낼 수 있는 강력한 기술이지만, 2020년 8월 현재 Amazon Review full에 대한 감성 분류의 [SOTA 모델](https://paperswithcode.com/sota/sentiment-analysis-on-amazon-review-full)의 정확도가 65.83일 정도로 다른 인공지능 분야에 비해 연구 성과가 적은 분야이기도 하다.  
게다가 현재까지 진행된 감성 분석 연구는 대부분 세계 공통어인 영어를 중심으로 이루어져 있어, 상대적으로 비주류 언어인 한국어에 대한 감성 분석 연구는 매우 부족한 상황이고, 이 마저도 대부분이 긍정-부정을 분류하는 극성 감성 분석(polarity sentiment analysis)에 그쳐 있다. 그러나 인간의 감정은 단순히 긍정, 부정 두가지로 분류하기에는 너무나도 넓은 스펙트럼을 가지고 있다.   
이에 우리 emo:)Tale 팀은 1. 한국어 자연어 텍스트에 대해서, 2. 글에 드러난 [기쁨, 슬픔, 공포, 분노]의 4가지 다중 감성을 분류하는 프로젝트를 진행하여 한국어 자연어처리에서 보다 세밀한 감성 분석을 가능케하고자 한다.  

<br>


# 2. 프로젝트 수행 과정

## 2.1 (사전 시도) Valence-Arousal Model에 의거한 감성 분류
- github link : https://github.com/zoomina/Valence-Arousal_based_sentiment_analysis  
- NRC-VAD(lexicon) : https://saifmohammad.com/WebPages/nrc-vad.html
- emobank(corpus) : https://github.com/JULIELab/EmoBank

Valence-Arousal은 EEG, ECG, GSR 등의 생체신호를 바탕으로 하는 연구에 주로 활용되는 정서모델로, 모호할 수 있는 정서를 객관적인 지표로 표현할 수 있기에 사전 시도로 선택하였다. Valence는 정서가에 해당하는 축으로 HRV(Heart Rate Variability)를 측정하여 얻어지게 되는데, 낮은 HRV는 부정적인 정서를, 높은 HRV는 긍정적인 정서를 나타낸다. Arousal은 각성가로 GSR(Galvanic Skin Response)을 측정하여 얻어지게 되는데, 높은 GSR은 높은 각성가를, 낮은 GSR은 낮은 각성가를 나타낸다. 사용된 lexicon과 corpus 역시 측정된 생체신호를 바탕으로 label되었기에 정서에 대한 객관적인 지표로 간주하였다.  

Arousal은 기존의 감성분석에 가장 많이 활용되는 긍/부정에 해당하는 지표로 두 가지의 이점을 얻을 수 있다.  
1. 긍/부정 연구에 활용되는 많은 데이터를 재사용할 수 있다.
2. 하나의 차원을 늘려 다양한 정서를 분류할 수 있다.

해당 시도는 regression 문제로 접근하였고, 이는 측정 지표로서 mse를 사용하기 때문에 정확도를 확신하기 힘들어 이후 classification 문제로 전환하게 되었다.  

<br>


## 2.2 KoBERT를 활용한 한국어 자연어 감성 분류

### 2.2.1 데이터 수집 및 전처리

먼저 크롤링을 통해 100,499개의 한국어 트위터를 확보하였다. 이를 팀원끼리 분배, 레이블링을 실시하여 총 33,853개의 텍스트로 이루어진 한국어 트위터 감성 코퍼스를 구축하였다. 이후 감정 데이터 간의 양적 비대칭성 문제를 해결하고자 각 감정에 대한 키워드를 4개씩 선정하고, 각 키워드 당 1,000개의 텍스트를 추출하여 총 16,000개의 텍스트로 이루어진 학습 코퍼스를 구축하였다. 

인간의 감정이라는 모호한 개념을 다루고 있기 때문에 데이터 레이블링에 명확한 기준이 필요했다. 단일 정서를 분류하는 모델을 만들 예정이기 때문에 복합정서 표현이 들어간 텍스트는 제거하였고, 텍스트 내에서 자주 반복되는 표현은 하나로 통합하였다. 트위터 문장 사이에 단어처럼 사용하는 해시태그는 그 내용을 보존하기 위해 ‘#’ 태그만 제거하였고, 한국어 분석 모델이기에 영어, 일본어 등 외국어는 제거하였다. 이외에도 너무 많은 오타가 포함된 경우, 다른 단어임에도 단어 내의 키워드 글자 중복으로 선정된 텍스트 등을 제거하였다. 이상의 기준을 바탕으로 팀원 8명이 레이블링을 진행하였다.


### 2.2.2 모델 구축 및 학습

이상의 데이터를 바탕으로 SKTBrain에서 학습시킨 한국어용 BERT 모델, KoBERT를 사용하여 모델을 구축하였다.


### 2.2.3 모델 평가

각 감정에 대해 1개씩 키워드를 선정하여 크롤, 레이블 한 트위터 텍스트에 대해 성능 평가를 한 결과, 공포(ㄷㄷ)를 제외한 기쁨, 분노, 슬픔에 대해서 약 95%에 달하는 높은 분류 정확도를 보였다.

 

## 2.번외. 기타 시도  
비록 채택되지는 않았지만, 시도해봤던 여러 시도들

### 2.번외.1 HiDEx
github link : https://github.com/cyrus/high-dimensional-explorer  

cosine distance를 기반으로 데이터의 양을 늘릴 수 있는 알고리즘으로, label되지 않은 대량의 corpus와 적정량의 lexicon을 이용하여 대량의 lexicon을 생성해낼 수 있다. 정서 관련 lexicon은 한국어는 물론이고 영어에서도 그 양이 부족하기에 lexicon 기반 시도의 정확도를 높이기 위해 시도하고자 하였다. 다만, 대량의 corpus에 대해 앞 뒤로 각 5-gram을 바탕으로 cosine distance를 적용하는 만큼 거대한 컴퓨팅 파워가 필요하여 시도하지 못했다.  

### 2.번외.2 규칙기반 한국어 감성분석 모델
Rule-based 모델. 정확도가 0.3 수준에서 벗어나지 못해 실패

### 2.번외. 추가바람

##

<br>

# 3. 프로젝트 결과

![그림1](https://user-images.githubusercontent.com/49966189/91823152-c0e74e80-ec73-11ea-954e-a945d4fe55a1.png)
![그림2](https://user-images.githubusercontent.com/49966189/91823156-c17fe500-ec73-11ea-940f-7ed03038f710.png)

학습된 모델을 활용할 한가지 방안으로 일기 형태의 텍스트를 입력 받아, 그 텍스트가 담고 있는 감정을 그래프 형태로 출력하는 서비스를 구현하였다. 

<br>

# 4. 추가 시도 (예정) 


향후 모델의 성능 향상을 위해 중립 정서를 분류하는 알고리즘을 추가하여 단순한 정보만을 담고 있는 텍스트나 선정한 4가지 감정으로 분류하기 애매한 텍스트를 분류하는 알고리즘을 추가할 예정이다. 또한 각 감정에 대한 키워드의 종류를 늘려 더 많은 양의 데이터로 모델을 학습시킨다면 보다 높은 성능을 낼 수 있다는 것이 확인되었다. 모델 자체에 대해서도 문장 단위를 입력으로 받는 KoBERT가 아닌, 형태소 단위를 입력으로 받는 KorBERT와 comment를 전문으로 처리하는 KcBERT를 활용한다면 더 높은 분류 성능을 기대할 수 있을 것이다. 

### 4.추가시도.1 KcBERT
지금 코드 수정중입니다.

<br>

# 5. Reference  


Ekman(1992), “Are there basic emotion?”, Handbook of cognition and emotion, p45  
Fredrik Hallsmar & Jonas Palm(2016), “Multi-class Sentiment Classification on Twitter using an Emoji Training Heuristic”, Computer Science  
남민지, 이은지, 신주현(2015), “인스타그램 해시태그를 이용한 사용자 감정 분류 방법”, 멀티미디어학회논문지 18호, p1391  
황재원, 고영중(2010), “감정 단어의 의미적 특성을 반영한 한국어 문서 감정분류 시스템”, 정보과학 논문지:소프트웨어및응용 37호, p317  
Chris Westbury et al(2014), “Avoid violence, rioting, and outrage; approach celebration, delight, and strength: Using large text corpora to compute valence, arousal, and the basic emotions”, The Quarterly Journal of Experimental Psychology, Vol. 68 No. 8, p1599  
James A. Russell(1978), “Evidence of Convergent Validity on the Dimensions of Affect”, Journal of Personality and Social Psychology, Vol. 36 No. 10, p1152  
James A. Russell(1980), “A Circumplex Model of Affect”, Journal of Personality and Social Psychology, Vol. 39 No. 6, p1161  
Margaret M. Bradley & Peter J. Lang, “Affective Norms for English Words (ANEW): Instruction Manual and Affective Ratings”, NIMH Center for the Study of Emotion and Attention  
The NRC Valence, Arousal, and Dominance(NRC-VAD) Lexicon[Website], https://saifmohammad.com/WebPages/nrc-vad.html
JULIELab/EmoBank[Website], https://github.com/JULIELab/EmoBank  
Pytorch로 시작하는 딥러닝 입문[Website], https://wikidocs.net/64517  
kh-kim/simple-ntc[Website], https://github.com/kh-kim/simple-ntc  
SKTBrain/KoBERT[Website], https://github.com/SKTBrain/KoBERT  
Beomi/KcBERT[Website], https://github.com/Beomi/KcBERT  
어텐션 매커니즘과 tansformer(self-attention)[Website], https://medium.com/platfarm/%EC%96%B4%ED%85%90%EC%85%98-%EB%A9%94%EC%BB%A4%EB%8B%88%EC%A6%98%EA%B3%BC-transfomer-self-attention-842498fd3225  
