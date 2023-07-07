##### 풍수해 및 침수 예측 시뮬레이터 개발 `(건기원, 행안부, 선도, 경북대, 포항공대)`

---

        # 0차 테스트 (from gang)
                LSTM + KNN 
                

        # 1차 테스트 (#done)
                Auto Encoder + Regressor -> 단순 예측 (1,2,3,6시간 강우)
                        -> 시계열 정보 무시

        # 2차 테스트 (#on going)
                Attention + Regressor -> 12시간 뒤 침수 예측 
                        -> 시계열 정보 최대한 이용
                        -> 1시간 예측 모델을 재귀적으로 이용 (bootstraping)


        
        1. 모델 경량화 (실시간 예측 on server)
        2. 외수예측 / 내수예측 
                외수 : data 690 -> 250000 (5 sections per year)
                내수 : data 48  -> 78     (5 sections per year)
        
