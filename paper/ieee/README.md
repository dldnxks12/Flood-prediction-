### IEEE Acces / Letters draft

---

- Abstract
  
      전지구적인 기후 변화로 인한 침수 관련 피해가 급증함에 따라 침수 관련 피해를 예측하고 예방하기 위한 연구가 꾸준히 진행중이다.
      본 논문은 강우량에 따른 침수 범위와 깊이를 실시간으로 예측하는 모델을 제안한다.
      실험에 사용한 데이터는 실제 강우량과 그에 대응하는 침수 흔적도이며, 데이터 증강을 위해 시뮬레이션 기반 강우량과 침수 흔적도를 포함시켰다.
      본 연구를 통해 실시간으로 예상 침수 깊이와 범위 흔적도를 생성하여 큰 피해를 예방할 수 있을 것으로 기대된다. 

      # key word
      manifold learning / multi process / auto encoder / linear regression / real-time flood prediction
  
---

- Introduction

---

- Background


      Current Problems (기존 연구의 한계)
      Related Works (기존 연구의 한계를 풀기 위한 연구들)
      AE / RG

---

- Data

      강우량 / 침수 흔적도 데이터를 어떻게 얻었는지 

---

- Proposed Method

      개발한 모델은 주어진 특정 기간 내의 강우에 대한 침수 결과 흔적도를 patch-base로 latent vector 를 추출하고, 
      추출된 latent vector에 대해 강우 데이터를 linear regression 한다.
      추론 과정에서는 강우 데이터를 linear regression 모델에 통과시킨 후 decoding 하여 침수 흔적 예측도를 생성한다.
      이러한 간단한 방법을 통해 기존의 수치해석 및 통계 모델 보다 더 빠르게 추론을 수행하며 머신 러닝 기반 모델 보다 더 높은 정확도를 확보한다.
  
      AE + RG / VAE + RG
      System Architecture 


---

- Experiment / Result / Discussion 

      경안천 / 석남동 / 반월천 예측 결과
        - 평가지표
        - 모델 크기
        - 추론 속도 

      # Compare
      1. AE vs VAE
      2. Original Latent vs patch-based latent
      3. AE+RG vs ViT
      4. RF / KNN / LSTM+KNN  vs AE+RG

---

- Conclusion


---

- Reference
  
  
