# 입력값 불러오기
# 입력값 전처리
# 모델 불러오기
# 모델 예측 및 저장
# GEOTIFF로 수정

import cv2
import sys
import psycopg2 # SQL
import argparse
import pickle
import pandas as pd
import numpy as np
import prd_module  # prediction module
import rec_module # img reconstruction module
import geopandas as gpd # 지리정보용 pandas
from shapely.geometry import Point  # shaply.geometry : 지리데이터 간의 관계를 연산해주는 기능
from shapely.wkt import dumps       # shaply.wkt      : 도형의 정보를 알 수 있는 포멧
# shapely : version 2 이상 lgeos 없음 -> downgrade to version 1.8.5 -> works!
# shaply : 기하학적인 객체의 조작 및 분석을 위한 라이브러리

def zero_filling(arr):
    for i in range(len(arr)):
        index = np.where(arr[i, :] > 0.0001)
        data = arr[i, index].flatten()
        temp = np.concatenate((arr[i], data))
        temp = np.delete(temp, index)
        arr[i] = temp
    return arr

# xllcorner : 왼쪽끝 격자 위치
# yllcorner : 아래끝 격자 위치
city_incheon = {'ncols': 992, 'nrows': 782, 'xllcorner': 923552, 'yllcorner': 1943137, 'cellsize': 5, 'EPSG': 5179}

class ModuleModel:
    def __init__(self, file_path = None, data_type = None, pred_once = None):
        self.data_type  = data_type # default : river
        self.pred_once  = pred_once

        if data_type == 'city' or None:
            self.gis_info = city_incheon
        else:
            raise NotImplementedError

        # Load Rainfall data
        self.rains = self.load_input(file_path)

        # ------------------------------------------------ #
        # 시각화
        self.rains_x = np.load('./data/x_data.npy')[-1, :]
        self.rains_y = np.load('./data/y_data.npy')[-1, :]

        if self.pred_once:
            self.rains_y = np.expand_dims(self.rains_y, axis = 0)
        # ------------------------------------------------ #

    # 시각화
    def pred_visualize(self, args):
        pred, t = prd_module.prediction(self.rains_x, args.predict_once)
        print(f"time takes to predict : {t} second")
        rec_module.reconstruction(pred, args.predict_once , self.rains_y, visualize=args.visualize)

    def load_input(self, file_path):
        # TODO : 1. Exel에서 불러오기 2. DB에서 불러오기
        data = pd.read_excel(file_path, engine='openpyxl').fillna(0).to_numpy()

        if self.data_type == 'city':
            rains = data[:, 1:]
            rains = rains.astype(np.float32)
            rains = np.transpose(rains)
        else:
            raise NotImplementedError

        # Predict only one scenario at once
        if self.pred_once:
            rains = rains[0]

        self.rains = rains
        return rains

    # 침수 예측 | 출력 타입 : numpy array
    def predict(self, rains=None):
        if rains is None:
            rains = self.rains # 강우 데이터

        pred_flood, t = prd_module.prediction(rains, self.pred_once)
        print(f"time takes to predict : {t} second")
        pred_flood = rec_module.reconstruction(pred_flood, self.pred_once)

        self.result = pred_flood
        return pred_flood

    # 예측 결과를 GeoDataFrame 형태로 변환
    def convert_to_gpd(self, result_img=None, save=False):
        nrows     = self.gis_info['nrows']
        xllcorner = self.gis_info['xllcorner']
        yllcorner = self.gis_info['yllcorner']
        cell_size = self.gis_info['cellsize']
        if result_img is None:
            result_img = self.result

        if self.pred_once == True :
            data = []
            where = np.where(result_img > 0)

            dxs = [1, 1, -1, -1, 1]
            dys = [1, -1, -1, 1, 1]
            gid = 0

            i = 0
            for a, b in zip(where[0], where[1]): # 예측한 값들의 위치 x, y좌표 == a, b
                var = result_img[a][b] # 좌표값 넣기
                point_x = xllcorner + b * cell_size  # 실제 지도의 좌표
                point_y = yllcorner - a * cell_size + cell_size * nrows
                # this is center coord
                for dx, dy in zip(dxs, dys):
                    gid += 1
                    x = point_x + cell_size * 0.5 * dx
                    y = point_y + cell_size * 0.5 * dy
                    fid = int((gid - 1) // 5)
                    geom = Point(x, y)
                    data.append({'gid': gid, 'var': var, 'id': int(0), 'orig_fid': fid, 'point_x': x, 'point_y': y,
                                 'geometry': geom})

        else:
            case_len = result_img.shape[0]
            data = []
            for case_idx in range(0, case_len):
                print(case_idx)
                result_idx = result_img[case_idx]
                where = np.where(result_idx > 0)

                dxs = [1, 1, -1, -1, 1]
                dys = [1, -1, -1, 1, 1]
                gid = 0
                i = 0
                for a, b in zip(where[0], where[1]):  # 예측한 값들의 위치 x, y좌표 == a, b
                    var = result_idx[a][b]  # 좌표값 넣기
                    point_x = xllcorner + b * cell_size  # 실제 지도의 좌표
                    point_y = yllcorner - a * cell_size + cell_size * nrows
                    # this is center coord
                    for dx, dy in zip(dxs, dys):
                        gid += 1
                        x = point_x + cell_size * 0.5 * dx
                        y = point_y + cell_size * 0.5 * dy
                        fid = int((gid - 1) // 5)
                        geom = Point(x, y)
                        print(case_idx, gid)
                        data.append({'gid': gid, 'var': var, 'id': int(case_idx), 'orig_fid': fid, 'point_x': x,
                                     'point_y': y,
                                     'geometry': geom})

        self.gdf = gpd.GeoDataFrame(data)

        # crs  : coordinate reference system
        # EPSG : coordinate system
        __crs = 'epsg:' + str(self.gis_info['EPSG'])
        self.gdf.crs = __crs

        # Save the GeoDataFrame to a shapefile or any other format you prefer
        if save:
            self.gdf.to_file(f'model_output_Seoknam.shp')

    # AI model 출력을 DB server에 쓰기
    def write_server(self):
        # connect : PostgreSQL과 연결
        conn = psycopg2.connect(database="flood", user="js", password="dldnxks1!",
                                host="localhost", port="5432")

        # cursor : 조작을 위한 인스턴스
        cur = conn.cursor()

        # erase all data in public.top19
        cur.execute("TRUNCATE TABLE public.js_test3")

        # insert data into public.top19  # dumps(geom) -> Geometry를 WKT 문자열로 변환
        self.gdf['geometry'] = self.gdf['geometry'].apply(lambda geom: dumps(geom))
        insert_query = """
        INSERT INTO public.js_test3 (gid, var, id, orig_fid, point_x, point_y, geom) VALUES (%s, %s, %s, %s, %s, %s, ST_GeomFromText(%s))
        """
        cur.executemany(insert_query, self.gdf.values.tolist())

        conn.commit() # commit을 통해 실제 transaction 발생시키기 - 데이터 입력, 수정, 삭제의 경우 이 함수를 반드시 호출
        cur.close()   # disconnect
        conn.close()  # disconnect


    # DB 서버에 저장된 강우량을 가져옴
    def load_rain(self):
        # load rain data from server
        conn = psycopg2.connect(database="flood", user="js", password="dldnxks1!",
                                host="localhost", port="5432")
        cur = conn.cursor()

        # 'public.rain_real' 사용해도 됨.
        # scenario 골라서 사용 가능
        query = """
        SELECT * FROM public.rain_prediction WHERE scenario = 'js_test3'
        """
        cur.execute(query)
        rows = cur.fetchall() # 조회하기 : SELECT

        # set rows with only its 3rd value
        rows = [float(row[2]) for row in rows]

        # DB 서버에 저장된 강우량이 5~10분 단위 데이터면
        # self.rains = np.array(rows)로 강우량을 load해서 사용해도 됨.
        self.pred_rains = np.array(rows)

        cur.close()
        conn.close()

# AI 결과가 서버에 잘 저장되었는지 확인하기 위한 함수
def load_data_to_geodataframe():
    conn = psycopg2.connect(database="flood", user="js", password="dldnxks1!",
                            host="localhost", port="5432")

    query = "SELECT * FROM public.js_test3"
    gdf = gpd.read_postgis(query, conn, geom_col='geom')
    gdf.to_file('model_output_Seoknam.shp')

    # .shp : shapefile - 벡터방식으로 공간정보를 저장
    conn.close()


if __name__ == '__main__':
    # Set model parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='city')
    parser.add_argument('--input_path', type=str, default='./data/Rainfall_data.xlsx')
    parser.add_argument('--predict_once', default = False)
    parser.add_argument('--visualize', default=False)
    args = parser.parse_args()

    # 모델 불러오기
    model = ModuleModel(args.input_path, args.type, args.predict_once)

    # ------------------------------------------------ #
    # 시각화
    if args.visualize == True and args.predict_once == True:
        model.pred_visualize(args)
        sys.exit()
    # ------------------------------------------------ #

    # 예측
    prediction = model.predict()
    print("# -- Prediction done -- #")
    model.convert_to_gpd(prediction)
    print("# -- Conversion done -- #")
    model.write_server()
    print("# -- Server Write done -- #")

    #load_data_to_geodataframe()

    print('TASK END')