# 기능 별 modulization

# 입력값 불러오기
# 입력값 전처리
# 모델 불러오기
# 모델 예측 및 저장
# GEOTIFF로 수정

import cv2
import sys
import argparse
import pickle
import pandas as pd
import numpy as np

# 지리정보용 pandas
import geopandas as gpd

# shapely : version 2 이상 lgeos 없음 -> downgrade to version 1.8.5 -> works!
# shaply : 기하학적인 객체의 조작 및 분석을 위한 라이브러리
from shapely.geometry import Point  # shaply.geometry : 지리데이터 간의 관계를 연산해주는 기능
from shapely.wkt import dumps       # shaply.wkt      : 도형의 정보를 알 수 있는 포멧

# SQL
import psycopg2

def zero_filling(arr):
    for i in range(len(arr)):
        index = np.where(arr[i, :] > 0.0001)
        data = arr[i, index].flatten()
        temp = np.concatenate((arr[i], data))
        temp = np.delete(temp, index)
        arr[i] = temp
    return arr


river_gyeongan = {'ncols': 4448, 'nrows': 4704, 'xllcorner': 213304, 'yllcorner': 505670, 'cellsize': 3, 'EPSG': 5186}
# city_incheon = {'ncols': 992, 'nrows': 782, 'xllcorner': 923552, 'yllcorner': 1943137, 'cellsize': 5, 'EPSG': 5179}
city_incheon = {'ncols': 992, 'nrows': 782, 'xllcorner': 167744.08196421713, 'yllcorner': 542825.5097635895, 'cellsize': 5, 'EPSG': 5186}


class ModuleModel:
    def __init__(self, model_path = None, file_path = None, data_type = None):
        self.pred_model = None
        self.data_type = data_type # river
        if data_type == 'river' or None:
            self.gis_info = river_gyeongan
        elif data_type == 'city':
            self.gis_info = city_incheon
        else:
            raise NotImplementedError

        self.load_model(model_path)
        self.rains = self.load_input(file_path)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.pred_model = pickle.load(f)

    def load_input(self, file_path):
        data = pd.read_excel(file_path, engine='openpyxl').fillna(0).to_numpy()

        if self.data_type == 'river':
            rains = data[1:, 1:]
            rains = rains.astype(np.float32)
            rains = np.transpose(rains)
            test_rains = zero_filling(rains)

        elif self.data_type == 'city':
            rains = data[:, 1:]
            rains = rains.astype(np.float32)
            rains = np.transpose(rains)

        else:
            raise NotImplementedError

        return rains

    # AI 모델로 침수 지역 예측.
    # output: numpy array
    def predict(self, rains=None):
        if rains is None:
            rains = self.rains # 690 x 36 강우량 데이터

        # predict only one scenario at once
        if len(rains.shape) > 1:
            rains = rains[4]
        # --------------------> rains : 1 x 36 --- 1개만 예측

        test_result = self.pred_model.predict(np.expand_dims(rains, axis=0)).flatten()

        if self.data_type == 'river':
            valid_index = np.loadtxt('./data/union_valid_index.txt', dtype=int)
            valid_index = (valid_index[0], valid_index[1])

            result_img = np.zeros((4704, 4448))
            result_img[valid_index] = test_result

        elif self.data_type == 'city':
            result_img = np.reshape(test_result, (782, 992))

        else:
            raise NotImplementedError

        self.result = result_img
        return result_img

    # 예측 결과를 GeoDataFrame 형태로 변환
    def convert_to_gpd(self, result_img=None, save=False, save_policy=None):

        # gis_info at river type : river_gyeongan = {'ncols': 4448, 'nrows': 4704, 'xllcorner': 213304, 'yllcorner': 505670, 'cellsize': 3, 'EPSG': 5186}
        nrows     = self.gis_info['nrows']
        xllcorner = self.gis_info['xllcorner']
        yllcorner = self.gis_info['yllcorner']
        cell_size = self.gis_info['cellsize']
        g_flag = False

        data = []
        # cell 수가 너무 많은 경안천 특별 처리
        if self.gis_info == river_gyeongan:
            g_flag = True
            data1 = []
            data2 = []
            data3 = []
            data4 = []

            # river에서만 하고 있는 점 유의. 나중에 city랑 river 나눠야 함.
            layer1 = gpd.read_file('./masking_layers/river_divide_1.shp')
            layer2 = gpd.read_file('./masking_layers/river_divide_2.shp')
            layer3 = gpd.read_file('./masking_layers/river_divide_3.shp')

        if result_img is None:
            result_img = self.result

        where = np.where(result_img > 0)

        dxs = [1, 1, -1, -1, 1]
        dys = [1, -1, -1, 1, 1]
        gid = 0

        if g_flag:
            for a, b in zip(where[0], where[1]):
                var = result_img[a][b]
                point_x = xllcorner + b * cell_size
                point_y = yllcorner - a * cell_size + cell_size * nrows
                # this is center coord
                for dx, dy in zip(dxs, dys):
                    gid += 1
                    x = point_x + cell_size * 0.5 * dx
                    y = point_y + cell_size * 0.5 * dy
                    fid = int((gid - 1) // 5)
                    geom = Point(x, y)

                    # shaply.geometry : contains : 지리적으로 포함하고 있는지 여부
                    if layer1.contains(geom).any():
                        data1.append({'gid': gid, 'var': var, 'id': fid, 'orig_fid': fid, 'point_x': x, 'point_y': y,
                                     'geometry': geom})
                    elif layer2.contains(geom).any():
                        data2.append({'gid': gid, 'var': var, 'id': fid, 'orig_fid': fid, 'point_x': x, 'point_y': y,
                                      'geometry': geom})
                    elif layer3.contains(geom).any():
                        data3.append({'gid': gid, 'var': var, 'id': fid, 'orig_fid': fid, 'point_x': x, 'point_y': y,
                                      'geometry': geom})
                    else:
                        data4.append({'gid': gid, 'var': var, 'id': fid, 'orig_fid': fid, 'point_x': x, 'point_y': y,
                                     'geometry': geom})

            self.gdf = gpd.GeoDataFrame(columns=['gid', 'var', 'id', 'orig_fid', 'point_x', 'point_y', 'geometry'])
            if str(1) in save_policy:
                self.gdf = self.gdf.append(data1)
            if str(2) in save_policy:
                self.gdf = self.gdf.append(data2)
            if str(3) in save_policy:
                self.gdf = self.gdf.append(data3)
            if str(4) in save_policy:
                self.gdf = self.gdf.append(data4)

        else:
            for a, b in zip(where[0], where[1]):
                var = result_img[a][b]
                point_x = xllcorner + b * cell_size
                point_y = yllcorner - a * cell_size + cell_size * nrows
                # this is center coord
                for dx, dy in zip(dxs, dys):
                    gid += 1
                    x = point_x + cell_size * 0.5 * dx
                    y = point_y + cell_size * 0.5 * dy
                    fid = int((gid - 1) // 5)
                    geom = Point(x, y)
                    data.append({'gid': gid, 'var': var, 'id': fid, 'orig_fid': fid, 'point_x': x, 'point_y': y,
                                 'geometry': geom})

            self.gdf = gpd.GeoDataFrame(data)

        # Save the GeoDataFrame to a shapefile or any other format you prefer
        if save:
            self.gdf.to_file(f'model_output_{self.data_type}.shp')

    # AI model 출력을 DB server에 쓰기
    def write_server(self):

        # connect : PostgreSQL과 연결
        conn = psycopg2.connect(database="flood", user="js", password="dldnxks1!",
                                host="localhost", port="5432")

        # cursor : 조작을 위한 인스턴스
        cur = conn.cursor()

        # erase all data in public.top19
        cur.execute("TRUNCATE TABLE public.top19")  # - 테이블에 입력된 데이터를 수정 (Update query)

        # insert data into public.top19
        self.gdf['geometry'] = self.gdf['geometry'].apply(lambda geom: dumps(geom)) # dumps(geom) -> Geometry를 WKT 문자열로 변환
        insert_query = """
        INSERT INTO public.top19 (gid, var, id, orig_fid, point_x, point_y, geom) VALUES (%s, %s, %s, %s, %s, %s, ST_GeomFromText(%s))
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
        SELECT * FROM public.rain_prediction WHERE scenario = 'top1'
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

    query = "SELECT * FROM public.top19"
    gdf = gpd.read_postgis(query, conn, geom_col='geom')

    # .shp : shapefile - 벡터방식으로 공간정보를 저장
    gdf.to_file('model_output_server_tester.shp')

    conn.close()

def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='river') # type : river | city

    parser.add_argument('--model_path', type=str, default='Full-KNN10-uniform_rivers.pkl')

    parser.add_argument('--input_path', type=str, default='./data/Hyetograph_of_Rainfll_Scenario_20221026.xlsx')

    parser.add_argument('--gyeongan_branch', type=str, default="1234")

    return parser.parse_args()

import matplotlib.pyplot as plt
if __name__ == '__main__':

    args = get_parameters()
    model = ModuleModel(args.model_path, args.input_path, args.type)

    result = model.predict()

    # cv2.namedWindow("title", cv2.WINDOW_NORMAL)
    # cv2.imshow('title', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    model.convert_to_gpd(result, save_policy=args.gyeongan_branch)
    model.write_server()
    load_data_to_geodataframe() # ok
    print('Done well!')
