# 입력값 불러오기
# 입력값 전처리
# 모델 불러오기
# 모델 예측 및 저장
# GEOTIFF로 수정

import psycopg2 # SQL
import argparse
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
river_gyeongan = {'ncols': 4448, 'nrows': 4704, 'xllcorner': 213304, 'yllcorner': 505670, 'cellsize': 3, 'EPSG': 5186}

class ModuleModel:
    def __init__(self, model_path = None, file_path = None, data_type = None, pred_once = None):
        self.pred_model = None
        self.data_type  = data_type # default : river
        self.pred_once  = pred_once

        if data_type == 'river' or None:
            self.gis_info = river_gyeongan
        else:
            raise NotImplementedError

        # Visualize Test
        self.rains_x = np.load('../Outer/xs_data.npy')[-15, :]
        self.rains_y = np.load('../Outer/ys_data.npy')[-15, :]

        if self.pred_once:
            self.rains_y = np.expand_dims(self.rains_y, axis = 0)
        print(self.rains_x.shape, self.rains_y.shape)


        # train-test data split
        # x_test  = np.load('xs_data.npy')[-30:, :] # 30 x 36
        # y_test  = np.load('ys_data.npy')[-30:, :] # 30 x 231432

        # Exel file
        # self.rains = self.load_input(file_path)

    # Visualize Result
    def pred_visualize(self, args):
        pred, t = prd_module.prediction(self.rains_x, self.rains_y, args.predict_once)
        print(f"time takes : {t}s")
        rec_module.reconstruction(pred, self.rains_y)

    def load_input(self, file_path):
        # TODO : DB에서 불러오기
        data = pd.read_excel(file_path, engine='openpyxl').fillna(0).to_numpy()

        if self.data_type == 'river':
            rains = data[1:, 1:]
            rains = rains.astype(np.float32)
            rains = np.transpose(rains)
            rains = zero_filling(rains)
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

        pred_flood = self.pred_model.predict(np.expand_dims(rains, axis=0)).flatten()

        # Masking task - predict over valid index only
        if self.data_type == 'river':
            valid_index = np.loadtxt('./data/union_valid_index.txt', dtype=int)
            valid_index = (valid_index[0], valid_index[1])

            flood_img = np.zeros((4704, 4448))
            flood_img[valid_index] = pred_flood

        else:
            raise NotImplementedError

        self.result = flood_img
        return flood_img

    # 예측 결과를 GeoDataFrame 형태로 변환
    def convert_to_gpd(self, result_img=None, save=False, save_policy=None):
        nrows     = self.gis_info['nrows']
        xllcorner = self.gis_info['xllcorner']
        yllcorner = self.gis_info['yllcorner']
        cell_size = self.gis_info['cellsize']

        data = []
        if result_img is None:
            result_img = self.result

        where = np.where(result_img > 0)

        dxs = [1, 1, -1, -1, 1]
        dys = [1, -1, -1, 1, 1]
        gid = 0

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
                data.append({'gid': gid, 'var': var, 'id': fid, 'orig_fid': fid, 'point_x': x, 'point_y': y,
                             'geometry': geom})

            self.gdf = gpd.GeoDataFrame(data)

            # crs  : coordinate reference system
            # EPSG : coordinate system
            __crs = 'epsg:' + str(self.gis_info['EPSG'])
            self.gdf.crs = __crs


        # Save the GeoDataFrame to a shapefile or any other format you prefer
        if save:
            self.gdf.to_file(f'model_output_{self.data_type}.shp')


    # AI model 출력을 DB server에 쓰기
    def write_server(self):
        # connect : PostgreSQL과 연결
        conn = psycopg2.connect(database="flood", user="Outer", password="dldnxks1!",
                                host="localhost", port="5432")

        # cursor : 조작을 위한 인스턴스
        cur = conn.cursor()

        # erase all data in public.top19
        cur.execute("TRUNCATE TABLE public.top19")

        # insert data into public.top19  # dumps(geom) -> Geometry를 WKT 문자열로 변환
        self.gdf['geometry'] = self.gdf['geometry'].apply(lambda geom: dumps(geom))
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
        conn = psycopg2.connect(database="flood", user="Outer", password="dldnxks1!",
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
    conn = psycopg2.connect(database="SUNDO", user="Outer", password="dldnxks1!",
                            host="localhost", port="5432")

    query = "SELECT * FROM public.top19"
    gdf = gpd.read_postgis(query, conn, geom_col='geom')

    # .shp : shapefile - 벡터방식으로 공간정보를 저장
    gdf.to_file('model_output_server_tester.shp')
    conn.close()


if __name__ == '__main__':
    # Set model parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='river')  # type : river | city
    parser.add_argument('--model_path', type=str, default='./Full-KNN10-uniform_rivers.pkl')
    parser.add_argument('--input_path', type=str, default='./Hyetograph_of_Rainfll_Scenario_20221026.xlsx')
    parser.add_argument('--gyeongan_branch', type=str, default="1234")
    parser.add_argument('--predict_once', default = True)
    args = parser.parse_args()

    # 모델 불러오기
    model = ModuleModel(args.model_path, args.input_path, args.type, args.predict_once)

    # ------------------------------------------------ #
    # 예측 후 시각화
    model.pred_visualize(args)
    # ------------------------------------------------ #

    # 예측
    # prediction = model.predict()
    # model.convert_to_gpd(prediction, save_policy=args.gyeongan_branch)
    #
    # # 서버 데이터 업데이트
    # model.write_server()
    # load_data_to_geodataframe()

    print('TASK END')