"""

*필요한 소스 코드 및 폴더
1. data dir      - 데이터 폴더
2. phase1 dir    - 학습 파라미터 폴더 1
3. phase2 dir    - 학습 파라미터 폴더 2
4. main.py       - 실행 모듈
5. network.py    - 학습 네트워크 모듈
6. prd_module.py - 예측 모듈
7. rec_module.py - 최종 결과 도출 모듈
# ------------------------------------------------- #

참고 / 주의하실 부분 TODO 표시해두었습니다.

# ------------------------------------------------- #
* argument 사용법

1. 예측된 값을 시각적으로 확인하고 싶을 경우 :
argument : visualize == True  | predict_once == True
----- + TODO 시각화 라고 표시된 주석들 해제해주시면 됩니다.
----- + x_data.npy , y_data.npy 데이터를 기반으로 수행됩니다.

2. 하나의 강우량 데이터에 대해 예측을 수행하고 서버에 올릴 경우 :
argument : visualize == False | predict_once == True


3. 다수의 강우량 데이터에 대해 예측을 수행하고 서버에 올릴 경우 :
argument : visualize == False | predict_once == False

4. db에서 데이터를 불러올 경우     | excel에서 데이터를 불러올 경우
argument : load_data == 'db' | load_data == 'excel'

"""

import sys
import time
import psycopg2
import argparse
import pandas as pd
import numpy as np
import prd_module
import rec_module
import geopandas as gpd
from shapely.geometry import Point
from shapely.wkt import dumps

def zero_filling(arr):
    for i in range(len(arr)):
        index = np.where(arr[i, :] > 0.0001)
        data = arr[i, index].flatten()
        temp = np.concatenate((arr[i], data))
        temp = np.delete(temp, index)
        arr[i] = temp
    return arr

city_incheon = {'ncols': 992, 'nrows': 782, 'xllcorner': 923552, 'yllcorner': 1943137, 'cellsize': 5, 'EPSG': 5179}

class ModuleModel:
    def __init__(self, file_path = None, data_type = None, pred_once = None, load_data = 'excel'):
        self.data_type  = data_type
        self.pred_once  = pred_once
        self.load_data  = load_data

        if data_type == 'city' or None:
            self.gis_info = city_incheon
        else:
            raise NotImplementedError

        # TODO : 데이터 path 관련 부분입니다. --- 최하단 argument를 변해주세요.
        if self.load_data == 'excel': # load from excel
            self.rains = self.load_input(file_path)
        else:
            self.load_rain() # load from db


        # ------------------------------------------------ #
        # TODO : 시각화를 위해 주석 해제해주시면 됩니다. --- 최하단 argument를 변경해주세요.
        # self.rains_x = np.load('data/x_data.npy')[-1, :]
        # self.rains_y = np.load('data/y_data.npy')[-1, :]
        #
        # if self.pred_once:
        #     self.rains_y = np.expand_dims(self.rains_y, axis = 0)
        # ------------------------------------------------ #

    # 시각화 함수
    def pred_visualize(self, args):
        pred  = prd_module.prediction(self.rains_x, args.predict_once)
        rec_module.reconstruction(pred, args.predict_once, self.rains_y, visualize=args.visualize)

    def load_input(self, file_path):
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
            rains = self.rains

        pred_flood = prd_module.prediction(rains, self.pred_once)
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
            for a, b in zip(where[0], where[1]):     # 예측한 값들의 위치 (x, y) 좌표 == (a, b)
                var = result_img[a][b]               # 좌표에 해당 하는 값
                point_x = xllcorner + b * cell_size
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
                result_idx = result_img[case_idx]
                where = np.where(result_idx > 0)

                print(np.max(result_idx))
                print()
                continue
                dxs = [1, 1, -1, -1, 1]
                dys = [1, -1, -1, 1, 1]
                gid = 0
                i = 0
                for a, b in zip(where[0], where[1]):     # 예측한 값들의 위치 (x, y) 좌표 == (a, b)
                    var = result_idx[a][b]               # 좌표에 해당 하는 값
                    point_x = xllcorner + b * cell_size
                    point_y = yllcorner - a * cell_size + cell_size * nrows
                    # this is center coord
                    for dx, dy in zip(dxs, dys):
                        gid += 1
                        x = point_x + cell_size * 0.5 * dx
                        y = point_y + cell_size * 0.5 * dy
                        fid = int((gid - 1) // 5)
                        geom = Point(x, y)
                        data.append({'gid': gid, 'var': var, 'id': int(case_idx), 'orig_fid': fid, 'point_x': x,
                                     'point_y': y,
                                     'geometry': geom})

        self.gdf = gpd.GeoDataFrame(data)

        __crs = 'epsg:' + str(self.gis_info['EPSG'])
        self.gdf.crs = __crs

        # Save the GeoDataFrame to a shapefile or any other format you prefer
        if save:
            self.gdf.to_file(f'model_output_Seoknam.shp')

    # AI model 출력을 DB server에 쓰기
    def write_server(self):
        conn = psycopg2.connect(database="flood", user="js", password="dldnxks1!",
                                host="localhost", port="5432")

        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE public.js_test3")

        # insert data into public.top19  | dumps(geom) --- Geometry to WKT
        self.gdf['geometry'] = self.gdf['geometry'].apply(lambda geom: dumps(geom))
        insert_query = """
        INSERT INTO public.js_test3 (gid, var, id, orig_fid, point_x, point_y, geom) VALUES (%s, %s, %s, %s, %s, %s, ST_GeomFromText(%s))
        """
        cur.executemany(insert_query, self.gdf.values.tolist())
        conn.commit()
        cur.close()
        conn.close()

    # DB 서버에 저장된 강우량을 가져옴
    def load_rain(self):
        # load rain data from server
        conn = psycopg2.connect(database="test", user="js", password="dldnxks1!",
                                host="localhost", port="5432")
        cur = conn.cursor()


        case_num  = 6    # TODO : 예측할 시나리오의 개수를 넣어주시면 됩니다.
        rain_data = []
        for idx in range(0, 6):
            query = "SELECT * FROM public.tb_tbed002_raininfo WHERE idx = '{}' ".format(idx+1)

            cur.execute(query)
            rows = cur.fetchall()

            # set rows with only its 3rd value
            rows = [float(row[3]) for row in rows]

            """
                학습 데이터 모양이 36개로 들어가야해서 가장 첫 번째 값을 버렸습니다. 
                이부분 참고해주시면 감사하겠습니다.
            """

            rain_data.append(rows[1:])

        self.rains = np.array(rain_data)

        cur.close()
        conn.close()

# AI 결과가 서버에 잘 저장되었는지 확인하기 위한 함수
def load_data_to_geodataframe():
    conn = psycopg2.connect(database="flood", user="js", password="dldnxks1!",
                            host="localhost", port="5432")

    query = "SELECT * FROM public.js_test3"
    gdf = gpd.read_postgis(query, conn, geom_col='geom')
    gdf.to_file('model_output_Seoknam.shp')
    conn.close()


if __name__ == '__main__':
    # Set model parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='city')
    parser.add_argument('--input_path', type=str, default='./data/Rainfall_data.xlsx')
    parser.add_argument('--predict_once', default = False)
    parser.add_argument('--visualize', default=False)
    parser.add_argument('--load_data', default='db') # TODO :  load data --- 'excel' or 'db'
    args = parser.parse_args()

    # 모델 불러오기
    model = ModuleModel(args.input_path, args.type, args.predict_once, args.load_data)

    # ------------------------------------------------ #
    # TODO : 시각화를 수행하려면 augment --visualize    == True로 변경해주세요.
    # TODO : 시각화를 수행하려면 augment --predict_once == True로 변경해주세요.
    # TODO : 예측된 값에 대한 시각화만 수행되며, 서버에 데이터를 올리지 않습니다.
    if args.visualize == True and args.predict_once == True:
        visualize_time = time.time()
        model.pred_visualize(args)
        print(f"Visualize time : {time.time() - visualize_time}s")
        sys.exit()
    # ------------------------------------------------ #


    pred_time = time.time()
    prediction = model.predict()
    print(f"Prediction time : {time.time() - pred_time}s")
    convert_time = time.time()
    model.convert_to_gpd(prediction)
    print(f"Conversion time : {time.time() - convert_time}s")

    write_time = time.time()
    model.write_server()
    print(f"Server writing time : {time.time() - write_time}s")

    # load_time = time.time()
    # load_data_to_geodataframe()
    # print(f"Data loading time : {time.time() - load_time}s")

    print('# --- TASK END --- #')