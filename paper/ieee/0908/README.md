### K-PAR `230908` 

    Input  : Hyetograph (regressor)/ Inundation trace map on DEM (auto encoder-spatial input) 
    Output : Inundation trace map on DEM (Spot / Depth) 

# TODO

    1. Location (다대/도림)/ Data augmentation (noise). City only? -> 괜찮을 것 같다.  

        test data 10 -> 20 -> 30 -> 40 으로 늘려가면서 PAR / AR / KNN / RF 비교
            -> 다른 지역 데이터 ? 각각 다른 feature의 데이터 --- 모델 성능 검증 가능  
            -> AE/VAE/CGAN 비교 실험은 같이 하되, 정 안되겠다싶으면 아예 빼버리기.
            -> CGAN은 왠만하면 비교군으로 놔두어야할 것 같은데, AE vs VAE해서 좋은 걸 main으로 잡기 

    2. rainfall 시각화 with Hyetograph

    3. architecture figure : auto encoder / fc layer 

    4. specify experiment environment. e.g., python version, torch version, cpu, gpu 

    5. KNN Masking 과 Threshold above 0.3m 비교 (KNN Masking 안하면 훨씬 가볍다.) 
        ref. -> A machine learning approach for forecasting and visualizing flood inundation information.

    6. (True - KPAR) --- box-plot을 통한 error 관련 시각자료 추가.
 
    7. Ensemble 사용? methods for reducing uncertainties and preventing overfitting. 
    

# Usable words

    #1 Promising accuracy (조심스럽게 얘기할거면)

    #2 We aim to replicate the output of a flood simulator. (simulator)

    #3 Pluvial flood : occurs when the amount of rainfall exceeds the capacity of urban storm water drainage system
    or the ground to absord it.

    #4 U-FLood / CAE -> 모두 raw spatial input image를 여러 feature로 나눠서 넣어준다.
        - U-FLOOD는 CAE 같이 spatial input을 여러 feature로 나눠서 넣어주지만, 어떤 feature를 넣는게 최선인지 확인
        - U-FLOOD는 CAE 대신 Skip-Connection을 이용한 U-NET 이용. 
        
        - K-PAR -> raw spatial grid를 사용한다. -> dataset을 준비하는게 간단하고 따라서 학습 접근성이 좋다. 

# Motivation 

현재 급격한 기후 변화로 인해 100년빈도, 300년빈도, 500년빈도 강우가 매년 오고 있다. 

더이상 예전의 방식대로 강우 강도를 추정하거나 오래된 자료를 사용하는 건 신뢰도 측면에서 생각해보았을 때 지양해야한다.

기존의 AI 모델은 제 성능을 내기 위해서 충분히 많은 데이터를 모아, 데이터와 침수 결과 간의 패턴을 학습하고 예측했다.

반대로 말하면, 데이터가 충분하지 않다면 모델의 예측 정확성을 확신할 수 없다는 이야기이다.

우리는 급격한 기후 변화에 발맞춘 합리적인 예측을 위해 비교적 최근 데이터만을 이용해서 학습시키고, 

또한 지속적으로 변화하는 환경에 맞는 새로운 데이터를 빠르게 모아서 학습해야한다.

일반적으로 데이터는 실제 침수 또는 시뮬레이션을 통해 확보한다.

하지만 실제 침수로 얻게 되는 데이터는 그 수가 매우 적고 시간이 오래 걸리기에 대부분 시뮬레이터를 이용하게 된다.

시뮬레이터 또한 변화하는 환경에 맞게 지속적으로 물리 역학적 요소들을 변경해야하며, 더 정확한 예측을 위해 물리 요소들을 추가할수록 

비선형 방정식을 풀어내는데 많은 시간이 소요된다. 약 15m-2h.

문제는 많은 노력과 시간을 투자해서 데이터를 쌓아 AI 모델을 학습시킨다고 한들, 환경적 요인이 지속적으로 바뀌면 시뮬레이터를 업데이트해야하고,

입출력 데이터의 특성이 바뀌기에 이에 맞는 데이터를 새로 모아 AI 모델을 다시금 학습시켜야한다.

본 논문은 이러한 근본적인 AI 기반 모델의 문제점에 초점을 맞추고, 적은 양의 데이터를 이용함에도 충분한 성능을 내도록 하는 것에 집중한다.

그렇게 된다면, 시뮬레이터를 통해 최소한의 데이터만을 만들어내고 이를 이용하여 빠르게 AI 모델을 학습시키고 예측에 사용할 수 있을 것이다. 


# Abstract Starts 

    # Data-driven flood emulation style.

    Computational complexity has been the bottleneck for applying physically based simulations in large
    urban areas with high-resolutions for efficient and systematic flooding analyses and risk assessment. 
    To overcome the issue of long computational time and accelerate thje prediction process, this paper
    proposes that the prediction of maximum water depth can be considered an image-to-image translation problem
    in which water depth rasters are generated using the information learned from data instead of by conducting simulations.

    # A deep convolutional neural netowrk style.

    Most of the 2D hydraulic/hydrodynamic models are still computationally too demanding for real-time applications.
    In this paper, an innovative modeling approach based on a CNN is presented for rapid prediction of fluvial flood inundation.
    The CNN model is trained using outputs from a 2D hydraulic model to predict water depths. 
    The performance of the CNN model is further comfirmed by benchmarking against a SVR metehod. 
    The results show that the CNN model outperforms SVR by a large margin. 
    The proposed CNN method offers greate potential for real-time flood forecasting considering its simplicty, performance and computational efficiency.

    # An ensemble neural ~ style.
    
    The model is successfully tested for spatially uniformly distributed synthetic rain events.
    The computational time of the model in the order of seconds and the acc of the results of the results are convincing.
    It suggest that the method may be useful for real-time forecasts. 


# Introduction

    # Data-driven flood emulation style.

    The combination of rapid urbanization and the rainfall intensity increase due to climate change is posing
    great challenges for flood risk management. Fast flood prediction methods are required to conduct systematic analysis
    and investigations of different urban planning and climate change scenarios.
    Furthermore, if rapid urban pluvial flood predictors are integrated with high temporal resolution online rainfall
    forecast services, it will be possible to inform citizens of likely urban pluvial flooding in advance, so that
    precautionary measures can be taken. 
    
    But, the current bottleneck for rapid urban pluvial flood analysis is the long computational time required by
    physically based simulation models. This problem becomes extremely significant for large simulation areas with 
    high spatial resolutions (or small raster grid size). 


    # A deeo convolutional neural network style.
    
    2D hydraulic/hydrodynamic models have been widely applied to simulate complex hydrological processes and flood dynamics. 
    Recent advancements in computing technology along with the increasing availability of high-resolution remotely sensed data, 
    such as terrain elevation and river morphology, have enabled these sophisticated models to be applied at the regional to global scales.
    However, due to their high computational demand, it is still challenging to use these physically-based sophisticated models
    for operational real-time flood forecasting.

    변화하는 Data 관련 :
    Furthermore, due to changing environment, e.g., land-use chagne, geomorphological change and engineering construction,
    , the flood scenarios may become outdated and new simulations are required to regulary update the database, 
    createing extra effort and resources for maintenance.

    
    # Development and Comparison of Two Fast Surrogate Models style. 

    However, such set up is generally too computationally intensive for real-time applications such as urban pluvial flood
    forecasting or for probabilistic approaches that require multiple simulations. 
    Two broad families of surrogate models can be considered:
    Lower-fidelity models, which are simplified physically based models preserving the main body of processes modeled in original system.
    And data-driven models which emulate the original model responses without analyzing the physical processes involved. 

    The physically-based models are being more and more complemented by data-driven models, which use ML methods
    to approximate the response of the original model. 
    In the water resources field, ANN are a popular approximation technique. They have been widely and successfully applied
    for modelling rainfall-runoff processes, forecasting streamflow and approximating rating curves, but applications 
    to urban hydraulic and hydrology are still scarce. Recent studies have reported significant spped gains over
    conventional hydrodynamic models with an acceptable loss of acc for most intended applications, which highlights
    the potential of this approach for predicting urban flooding in real-time. 

    However! the generalization ability beyond the training data is a concern when applying this type of models.


    # An ensemble neural style.

    Urbanization may have a significant impact on the rainfall runoff relation.
    Related decrease of pervious area may increase urban floods. Mainly short term precipitation events with 
    high intensity can lead to an exceedance of the capacity of drainage systems and flooding of the surface. 
    With climate change, those events will occur more often in the future and the monetary and the non-monetary damage 
    may increase significantly. For the implementation of early warning systems, fast forecast models are indispensable.
    Such models include the rain forecasting and the prediction of inundated areas. 


# Used models / Simulator / Data / Metric 

    # A deeo convolutional neural network style.

    To assess the capability of the proposed CNN model in emulating the results of a 2D hydraulic model,
    the DL predictions in terms of water depths are directly compared with the outputs from the adopted hydraulic model.
    In addition, the performance of the CNN model will be further evalutated by comparing with an SVR approach, 
    one of the popular ML methods used in earlier studies.

        # Comparison
        -> This paper use point-wise comparison between the models(CNN vs Hydraulic) at 18 pre-selected points.
        -> Our paper use point-wise comparion between models at all points!! 

    IT it worth noting that the primary objective of this study is to investigate the capability of the CNN model
    in emulating the outputs of a 2D hydrodynamic/hydrualic model. 
    Therefore, the outputs of the simulator rather than the field observations are used as the reference to assess the predictive performances of the ML models.

    A depth threshhold of 0.3m is applied to negate the insignificant depths from the target data.
    This value is selected based on the flood hazard thresholds for different objects as suggested by -.

        # TODO 
        -> KNN 사용하지 않고, Auto-Encoder만 사용한 결과에서 threshold 0.3m 적용해보자. 
        -> 작은 depth에 대해서는 KNN 안써도 괜찮을 수 있다.
    

    # An ensemble neural style.

    The purpose of the ANN is to provide a fast simplified model as substitute for a slow physically based numerical model.
    Therefore, we use flood simulations generated with a physically based numerical model as virtual true.
    The model results are used for training of the ANN as well as to assess the ANN as a method for flood prediction. 

        # valid grid
        Only cells which have the potential to be flooded, are of interest.
        To forecast flooding in a cell which is never flooded in the training dataset, the water level in that cell would
        have to be extrapolated, which is not a useful approach. 
        Thus cells which have a maximum depth of zero in every sample of the training set are neglected.
        This constraint reduces the amount neurons in the output layer.
    
        -> 우리는 100년/200년/300년 빈도 강우까지 다 해서 flood 가능성이 있는 grid들에 대해서 학습을 진행했다.

    # Overview of models being used

    1. Auto Encoder
    2. Linear Regression
    3. KNN
    4. CGAN / VAE : 선택

# Patch?  

    *Data-driven flood ~랑 K-PAR는 Patch 사용의 근본적인 이유가 다르다. 

    # Data-driven flood emulation style.

    0. Patch image 를 input 으로 
    1. 1개의 network만 학습 
    2. tranining dataset augmetnation 효과
    3. CAE input image size 줄이면 학습이 더 잘된다.
    4. 각 catchment에 대해 patch 화해서 10000개 x 18의 학습 데이터 보유
    

    # K-PAR 

    0. patch grid만 input으로 
        - input size를 극단적으로 줄이기 위함 (river -> possible inundation range / city -> past pluvial flood)
        - simulation 으로 얻은 결과를 처리하는 시간도 단축 (rainfall + inundation trace map only)
        - patch image를 사용하게 되면, no-flood patch의 양이 많아져서 data-imbalance 문제가 발생
    1. 200개의 diffent network 사용     
        - 1개 network를 여러 patch에 대해 사용하지 않음 
        - 각 patch location 마다 가지는 특징이 있을텐데, 이게 generalization 되지 않도록 하기 위함.
    2. Curse of dimentionally 에 따라, dimension이 작아지면 학습에 필요한 data 수가 작아질 수 있음.
    3. AutoEncoder에 넣는 Input dim 작을수록 학습 성능 향상
    4. 1 catchment에 대해 40개의 학습 데이터 보유 

    # Choosing patch size -> trade-off between time and acc

    # About Patch and correlation ... in an ensemble paper~ 

    The subnets are independent, which implies that the correlation between cells with a short distance that belong to 
    different subnets get lost. However, the philosophy of a data-driven model is that every correlation is contained in the data
    and is therefore captured. This means the interconnection of the subnets is included in the data.

        -> ensemble paper 인용하면서 이야기하기.

    
