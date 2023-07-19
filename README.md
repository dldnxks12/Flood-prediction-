##### 풍수해 및 침수 예측 시뮬레이터 개발 `(건기원, 행안부, 선도, 경북대, 포항공대)`

---

- `dev-road`


        1.   load rainfall from DB server 
        2-1. predict 1 : predict flood map from rainfall data  
        2-2. predict 2 : predict rainfall in 10 min, and recursively use this for predict upto 3 hours.
        3.   save predicted flood map / flood data to DB server
  

---

- `test`

  
        # 0차 테스트 (from gang) : LSTM + KNN 
  
        # 1차 테스트 (#done) : AE + Regressor
                - 실제 강우량 기반 침수 예측 
                - 시계열 정보 무시
                - module화 완료
                  - model weight  : 270MB 
                  - sampling time : 6-7s 


        # 2차 테스트 (#done) : AE + Regressor / LSTM -> 1시간 뒤까지 강우 예측 ok
                - 예측 강우량 기반 침수 예측
                - 시계열 정보 이용

                                        
        # 3차 테스트 (#on going) : VAE + Regressor / transformer
                - 2차 모델 고도화
        

---

- `TODO `

        1. 모델 경량화 
        2. 외수예측 
        3. 내수예측 
                
        
