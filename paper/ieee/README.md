#### `IEEE Letters - Access`

---

- `Contribution`

    
        1. 깊이와 위치를 동시에 예측
        2. Patch-based
        3. 수치해석 모델 사용 x
        4. 내가 만든 dataset 으로 실험 

---

- Abstract

      # 왜 이걸하나?
  
          급격한 기후 변화로 인한 침수 관련 피해 발생 건수가 급증하고 피해 정도도 점점 더 강력해지고 있다.
          이에 따라 침수 위치와 피해량을 사전에 예측하고 예방하기 위한 연구가 활발히 진행중되고 있다.
          한 편, 침수는 non-linearity + dynamic nature 로 인해 예측하기 매우 어렵고 복잡한 현상이다.
  
      # 기존 방법과 한계 간단히 
      
          기존의 연구는 유체역학 (hydrodynamic) 기반 모델과 같은 수치해석적 접근법이나 통계적 접근법을 사용하였지만, 
          비선형 방정식을 푸는 등 예측에 많은 시간이 소요.
          이러한 단점은 국지성 호우 등 제한된 시간 및 지역에 발생할 수 있는 피해를 실시간으로 예측하는 데에 큰 장애물이 된다.
    
          최근 이러한 비선형적 예측에 좋은 성능을 보이는 머신 러닝 / 딥 러닝을 이용해 침수 피해를 예측하려는 연구가 주목받고 있다.

      # 제안한 방법이 어떤 이점이 있는지?? ---- 추가 
  
          본 논문은 실시간으로 강우량에 따른 침수 위치와 깊이를 예측하는 딥러닝 기반 모델을 제안한다.
          제안된 모델은 오롯이 딥러닝 모델만을 이용하여 침수 위치와 깊이를 실시간으로 예측하며, ~ 의 성능이 나온다. 


      본 연구를 통해 실시간으로 예상 침수 깊이와 위치 예상도를 생성하여 큰 피해를 예방할 수 있을 것으로 기대된다. 

      # key word
      manifold learning / multi process / auto encoder / linear regression / real-time flood prediction
  
---

- Introduction


      # flood 에 관심이 높아지고 있다.
      # flood가 어떤 damage 를 주는지
      # flood 의 예측이 왜 어려운지
        - river : soil property / land usage / climate / river basin / other geophysical elements.

      그래서 flood 를 미리 예측하고 예방하는게 매우 중요하다.
      이 문제를 해결하기 위한 common method - physical / statistical and 최근에는 ml+dl 

      # 기존 연구 및 한계 
      # - 기존 방법 physical / statistical - 장단점

        - physical :
  
      physical model 은 다양한 factor 간의 물리적 행동 양상과 그들 간의 상호 작용을 서술하는 수학적 방정식을 기반으로
      침수를 예측한다. 이 방법은 올바른 모델링을 기반으로 좋은 성능을 예상할 수 있으며,
      이를 통해 강우를 예측하거나 침수 깊이, 침수 위치를 예측하는 등의 역할을 수행할 수 있다.
      하지만 이러한 접근은 변화하는 환경에 맞추어 지속적으로 모델을 수정 개선해야하며, 예상치 못한 변수에 대해 취약하다.
      최근 simulator 의 발전으로 인해 다양한 수문학적(hydrological) 사건들의 prediction 이 향상되고 있다.
      그럼에도 불구하고 여전히 비선형적 방정식을 계산하는 과정에서 소요되는 긴 시간으로 인해 실시간 예측이 어렵다는 단점을 가지고 있다.

      한 편, 이러한 단점을 완화하기 위해 ML/DL 을 physical model과 hybrid로 사용하여 실시간성과 예측 accuracy 간의
      적절한 타협점을 찾는 방법들이 제안되고 있다.
  
          - statistical :
  
      statistical model 은 침수의 내재된 pattern 을 찾기 위해 historical data 를 이용하는 방법이다.
      MLR / ARIMA / LS-SVM
      근데, 이 모델들은 계절적인 변화 특성을 capture 하기 위해 매우 오랜 기간의 데이터가 필요하다.
      즉, 한 지역의 flood pattern 을 모델링하기 위해 해당 지역의 100년 혹은 그 이상의 데이터가 필요한 경우가 있다.
        

      # 기존 한계를 극복하기 위한 접근
  
        ML/DL 기반 방법은 위의 단점들을 극복할 수 있다.
        먼저, 비선형적 특성을 잘 파악하는 장점으로 인해, 복잡한 비선형 방정식을 풀지 않아도 된다.
        그로 인해 추론 시간이 대폭 줄어들어 실시간 예측이 가능하다.

        또한 neural network 의 빠르고 정확하게 data에 내재된 pattern 을 파악하는 능력으로,
        통계적 모델과 같이 매우 긴 기간 동안의 데이터가 필요하지 않다. 

        무엇보다 양질의 학습 데이터만 있다면 physical / statistical 기반 방법에 비해 domain knowledge 가 많이 필요하지도 않다.
  
        실제로 최근 풍부한 데이터를 기반으로한 ML/DL 방법은 physical / statistical 기반 방법보다 performance가 좋다. 

      # 그래서 침수 위치와 깊이 예측에 어떤 방법들이 제안되고 있는지 - related work

          - KNN / RF / CNN / Regression?

            - KNN / RF 그냥 쓰면 성능 X
            - Physical model 이랑 곁들여 쓰면 성능 O
  

      # 1. 요것들이 가지고 있는 문제가 무엇인가?  2. 그래서 내가 제안하는 방법이 어떤 면을 파고드는가?

            - only 실시간 강우 데이터를 기반으로 침수 흔적을 예측하지 않는다. 
  
      # 어떤 데이터를 사용헀고, 어디에 test 를 했는지?

          실험에 사용한 데이터는 실제 강우량과 그에 대응하는 침수 흔적도이며,
          데이터 증강을 위해 시뮬레이션 기반 강우량과 침수 흔적도를 포함시켰다.

          경안천 / 반월천 두 곳에 test

            - 왜 이 두 곳을 선택했는지 간단히 설명 
  


---

- Background


      Rapid forecasting - KNN / RF를 수치해석 모델과 겸해서 사용 - depth 예측 
      Jiwook            - KNN / RF 를 only data based로 사용   - depth + 위치 예측 
    

      RF / KNN / LSTM / CNN 
      AE / RG

---

- Data

      실험에 사용한 데이터는 실제 강우량과 그에 대응하는 침수 흔적도이며,
      데이터 증강을 위해 시뮬레이션 기반 강우량과 침수 흔적도를 포함시켰다.
  
      강우량 / 침수 흔적도 데이터를 어떻게 얻었는지

      # 경안천과 반월동에 대해 test
        - 경안천 / 반월동 사진 
  

      입력 데이터 : 10분 간격 1~6시간 / 1~12시간 분포의 강우 데이터
      출력 데이터 : 침수 깊이 및 위치 흔적도 3m x 3m / 5m x 5m 

      -> 강우 그래프 보여주기
      -> 침수 흔적도 보여주기

      침수된 적이 있는 pixel 만을 고려
      0~18m 정도의 침수 깊이 값들이 있으며, 통상적으로 1m 이상을 침수로 간주.
      

---

- Proposed Method

      개발한 모델은 주어진 특정 기간 내의 강우에 대한 침수 결과 흔적도를 patch-base로 latent vector 를 추출하고, 
      추출된 latent vector에 대해 강우 데이터를 linear regression 한다.
      추론 과정에서는 강우 데이터를 linear regression 모델에 통과시킨 후 decoding 하여 침수 흔적 예측도를 생성한다.
      이러한 간단한 방법을 통해 기존의 수치해석 및 통계 모델 보다 더 빠르게 추론을 수행하며 머신 러닝 기반 모델 보다 더 높은 정확도를 확보한다.

      일반적으로 Original image 를 그대로 사용하게 되면 학습에 불필요한 None 값까지 고려하게 된다.
      즉, 전체 4448x4704 pixel을 그대로 사용하게 된다.
      하지만, 유의미한 침수 지역만을 선별하여 (Valid index) 이를 patch-base로 학습하게 될 경우, 20만 pixel 정도로 대폭 줄일 수 있다. 

      Original image vs Patch-based image 간의 모델 크기 및 학습 속도, 성능은 experiment 에서 서술 

      AE + RG / VAE + RG
      System Architecture 


---

- Experiment / Result / Discussion 

      경안천 / 반월천 예측 결과

      # Compare
      1. AE vs VAE
      2. Original Latent vs patch-based latent - ablation study 
      3. AE+RG vs ViT
      4. RF / KNN / LSTM+KNN  vs AE+RG

        - 평가지표 결과 - 깊이 (RMSE / Max-Min error / MRE) - 위치 (Sensitivity / Precision / F1 / Accuracy / IoU)
        - 모델 크기
        - 추론 속도
        - 학습 속도

      
---

- Conclusion


    우리 침수 흔적 잘 예측한다. 그러니까 강우량만 주어진다면, 얼마든지 침수 피해를 예측하고 예방할 수 있다. 

---

- Acknowledge

---

- References
  
  
