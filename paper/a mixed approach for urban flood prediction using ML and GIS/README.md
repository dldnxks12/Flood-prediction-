### a mixed approach for urban flood prediction using ML and GIS

---

- `Summary`


        ML pipeline의 초기 2가지 문제 
        
                1. limited number of weather stations (기상관측소) -> 더 다양한 상황에 대해서는 강건하지 못함 (걍 데이터 부족)
                2. lack of representation of morphological factors(지형학적 특성)
                        -> 지역적 취약점들을 고려하지 못함
                

                The lack of high-resolution of topographic / hydrologic data (등고 / 수문학 정보)
                        -> 도심 지역 침수 모델링의 발전을 제한
                

        Catchment area / rainfall intensity 를 이용한 ML model 성능이 가장 좋았다.
                -> 하지만 model fitting에 제한된 수의 관측소를 사용할 때 robustness drops...
                        -> i.e, generalization is poor
                      

        So, ML pipeline에 더해 GIS 정보를 이용 (spatial representation 정보를 이용)
                -> GIS 정보를 이용해서, intrinsic geographic factor를 추가 
                        -> 특정 지역의 침수 가능성을 높이거나 낮출 수 있다.
                        
                        i.e) ML이 특정 부분에 침수가 난다고 예측했는데, 그에 대한 신뢰도를 GIS Model로 한 번 더 filtering 해준다고 생각
                        
                        Geographic information system :
                                지리정보를 컴퓨터 데이터로 변환해서 효율적으로 사용하기 위한 정보시스템 


        In conclusion, 이 논문은 ML + GIS로 침수 지역을 예측한다.
        
                ML  : catchment area / rainfall intensity ..
                GIS : spatial representation (topographic, hydrologic, grographic, ...)


---

- `Hot Spot Analysis`


        침수 발생은 특정 공간 Unit에 관련되어있기 때문에, GI*기반의 GIS를 써서 spatial cluster를 찾아냈다.
        이 spatial cluster의 significant가 높다면 Hot spot, otherwise cold spot. 

        So, Hot spot analysis는 local 취약점 및 침수 기준 공간적 구분을 하기 위한 reference로 잘 쓰였다.
        
        
---

- `Hot Spot Analysis + ML`


        Hot Spot Analysis -> 유용한 spatial pattern 찾기에 유용
        ML                -> 주어진 데이터로 future event에 대한 숨겨진 pattern 찾기에 유용
        
        
                -> But 독립적으로는 침수 예측 시스템을 만들기엔 부족 
                        -> ML + GIS 짬뽕 모델로 뽑은 점수를 기반으로 risk index를 계산하는 기술을 제안 
                
                
        Random Forest가 제일 성능이 준수 
        
        + Paper에 따르면, 침수 예측은 heterogeneous process로 진행되는게 맞다고 한다.
        
                -> Spatial info에 대한 과정 
                -> Non-spatial feature data를 이용한 과정 
        

        
