import os
import time
import cv2

import numpy as np
import pandas as pd

import tensorflow as tf

from model import JS08Settings

class LSTM_model():
    
    def __init__(self):
        
        # tfmodel use gpu
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
        # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            except RuntimeError as e:
                # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
                print(e)
            

        self.lstm_model = self.load_model()

        day = 24*60*60
        self.day_sin = np.sin(time.time() * (2 * np.pi / day))
        self.day_cos = np.cos(time.time() * (2 * np.pi / day))
        

        self.day_sin_mean = -0.0
        self.day_sin_std = 0.7073

        self.day_cos_mean = -0.0
        self.day_cos_std = 0.7073

        # self.prev_mean = 13.6326
        # self.prev_std = 3.271656

        self.prev_mean = 12.9587
        self.prev_std = 7.6698
        
        
        self.sun_alt_mean = 12.472478
        self.sun_alt_std = 35.432907
        
        self.sun_azm_mean = 181.121984
        self.sun_azm_std = 110.512035
        
        self.sun_yn_mean = 0.578326
        self.sun_yn_std = 0.493841
        
        self.mean_se = {'prev' : self.prev_mean, 'Day sin' : self.day_sin_mean , 'Day cos' : self.day_cos_mean, 
                        'sun_altitude' : self.sun_alt_mean, 'sun_azimuth' : self.sun_azm_mean, 'sun_yes_no' : self.sun_yn_mean}
        self.std_se = {'prev' : self.prev_std, 'Day sin' : self.day_sin_std , 'Day cos' : self.day_cos_std,
                       'sun_altitude' : self.sun_alt_std, 'sun_azimuth' : self.sun_azm_std, 'sun_yes_no' : self.sun_yn_std}
    
    def load_model(self):
        """" 모델을 불러오는 함수 """
        # 저장된 모델 경로 입력
        # model_path = "model_1654675043"
        model_path = "./saved_model_20230817150727_multi/my_model"
        # model_path = "./saved_model/my_model"
        # 모델 불러오기
        new_model = tf.keras.models.load_model(model_path, compile=False)
        new_model.trainable = False
        print(new_model.summary())

        # 모델 리턴
        return new_model
    
    def predict_visibility(self, epoch, input_df):
        
        # input_data = [(visibility - self.prev_mean) * self.prev_std, (self.day_sin - self.day_sin_mean) * self.day_sin_std, (self.day_cos - self.day_cos_mean) * self.day_cos_std]
        data_df = (input_df - self.mean_se) / self.std_se
        w2 = WindowGenerator(train_df=data_df, input_width=18, label_width=1, shift=1,
                    label_columns=['prev'])

        example_window = tf.stack([np.array(data_df[:w2.total_window_size])])
                                

        example_inputs, example_labels = w2.split_window(example_window)
        # input_data = tf.constant(input_data)
        # input_data = input_data[np.newaxis, np.newaxis, :]
        prediction = self.lstm_model(example_inputs)
        print(prediction.shape)
        prediction_cvt = (prediction.numpy() * self.prev_std  + self.prev_mean)
        
        del prediction
        return prediction_cvt

    def find_closest_value(self, number, values):
        closest_value = None
        min_diff = float('inf')

        for value in values:
            diff = abs(number - value)
            if diff < min_diff:
                min_diff = diff
                closest_value = value

        return closest_value


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df=None, val_df=None, test_df=None,
                label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                                enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
          # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

if __name__ == '__main__':
    
    import pandas as pd
    from datetime import timedelta
    df = pd.read_csv('D:/JS08/data/Prevailing_Visibility/2023/0606.csv')
    print(df.head())
    day = 24*60*60
    now = pd.Timestamp.now()
    current_time = now.timestamp()
    pv = 20
    
    lstm_model = LSTM_model()
    coulmns = ["prev", "Day sin", "Day cos"]
    lstm_df = pd.DataFrame(columns = coulmns)
    if lstm_df.shape[0] < 18:
            
            if lstm_df.shape[0] == 0:
                for i in range(18, 0, -1):
                    before_time = now - timedelta(minutes=10 * i)
                    # print("now : ", before_time)
                    new_serires = {'prev':  pv, "Day sin" : np.sin(before_time.timestamp() * (2 * np.pi / day)), "Day cos" : np.cos(before_time.timestamp() * (2 * np.pi / day))}
                    lstm_df = lstm_df.append(new_serires, ignore_index = True)                
                
            # if self.lstm_df.shape[0] == 0:
            #     before_time = now - timedelta(minutes=15)
            #     new_serires = {'prev':  pv, "Day sin" : np.sin(before_time.timestamp() * (2 * np.pi / day)), "Day cos" : np.cos(before_time.timestamp() * (2 * np.pi / day))}
            #     self.lstm_df = self.lstm_df.append(new_serires, ignore_index = True)

            new_serires = {'prev':  pv, "Day sin" : np.sin(current_time * (2 * np.pi / day)), "Day cos" : np.cos(current_time * (2 * np.pi / day))}
            lstm_df = lstm_df.append(new_serires, ignore_index = True)
            
    predict_cvt = lstm_model.predict_visibility(current_time, lstm_df)
    predict_vis = predict_cvt[0][-1][0]
    print("predict_vis", predict_vis)