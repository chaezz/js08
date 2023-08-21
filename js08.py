#!/usr/bin/env python3
#
# Copyright 2021-2022 Sijung Co., Ltd.
#
# Authors:
#     cotjdals5450@gmail.com (Seong Min Chae)
#     5jx2oh@gmail.com (Jongjin Oh)

import os
import sys
import vlc
import time
import ephem
import math

from datetime import timedelta, datetime
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Process, Queue

from PySide6.QtGui import QPixmap, QIcon, QPainter, QPen
from PySide6.QtWidgets import (QMainWindow, QWidget, QFrame, QMessageBox)
from PySide6.QtCore import (Qt, Slot, QRect,
                            QTimer, QObject, QDateTime)

from login_view import LoginWindow
from video_thread_mp import producer
from log_view import LogView
from js08_settings_admin import JS08AdminSettingWidget
from js08_settings_user import JS08UserSettingWidget
from model import JS08Settings
from curve_thread import CurveThread
from clock import clock_clock
from consumer import Consumer
from thumbnail_view import ThumbnailView
from save_log import log

from visibility_view import VisibilityView
from discernment_view import DiscernmentView

# UI
from resources.main_window2 import Ui_MainWindow

# Warning Message ignore
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# seongmin
from lstm_model_print import LSTM_model
from predict_visibility_view import Predict_VisibilityView, Vis_Chart


class JS08MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, q, _q):
        super(JS08MainWindow, self).__init__()

        login_window = LoginWindow()
        login_window.sijunglogo.setIcon(QIcon('resources/asset/f_logo.png'))
        login_window.exec()

        _p.start()

        self.setupUi(self)
        # self.showFullScreen()
        print(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

        self.get_date = []
        self.get_epoch = []
        self.q_list = []
        self.q_list_scale = 1440
        # seongmin
        self.p_list = []
        self.p_list_scale = 10
        self.result = pd.DataFrame

        current_time = time.strftime('%Y-%m-%d %H:%M:00', time.localtime(QDateTime.currentSecsSinceEpoch()))
        year = current_time[:4]
        md = current_time[5:7] + current_time[8:10]

        self.q_list = self.get_data(year, md)

        self._plot = VisibilityView(self, self.q_list_scale)
        self._polar = DiscernmentView(self)
        # self._predic_plot = Predict_VisibilityView(self, self.p_list_scale)
        self._predic_plot = Vis_Chart(self)
        self.view = None
        self.km_mile_convert = True

        self.visibility = None
        self.prevailing_visibility = None
        self.predict_vis_meter = None
        self.graph_visibility_value = []

        self.year_date = None
        self.data_date = []
        self.data_time = []
        self.data_vis = []

        self.front_video_widget = VideoWidget(self)
        self.front_video_widget.on_camera_change(JS08Settings.get('front_main'))

        self.rear_video_widget = VideoWidget(self)
        self.rear_video_widget.on_camera_change(JS08Settings.get('rear_main'))

        self.video_horizontalLayout.addWidget(self.front_video_widget.video_frame)
        self.video_horizontalLayout.addWidget(self.rear_video_widget.video_frame)

        self.graph_horizontalLayout.addWidget(self._plot)
        self.polar_horizontalLayout.addWidget(self._polar)
        # seongmin
        self.horizontalLayout_6.addWidget(self._predic_plot.chart_view)
        

        if JS08Settings.get('right') == 'administrator':
            self.log_view.setIcon(QIcon('resources/asset/log.png'))
        else:
            self.log_view.setEnabled(False)

        self.setting_button.setIcon(QIcon('resources/asset/settings.png'))
        self.setting_button.enterEvent = self.btn_on
        self.setting_button.leaveEvent = self.btn_off

        self.front_label.paintEvent = self.front_label_paintEvent
        self.rear_label.paintEvent = self.rear_label_paintEvent

        self.setWindowIcon(QIcon('logo.ico'))
        self.logo.setIcon(QIcon('resources/asset/f_logo.png'))
        self.time_button.setIcon(QIcon('resources/asset/clock.png'))
        self.timeseries_button_2.setIcon(QIcon('resources/asset/graph.png'))
        self.timeseries_button.setIcon(QIcon('resources/asset/polar.png'))
        self.prevailing_vis_button.setIcon(QIcon('resources/asset/vis.png'))
        self.button.setIcon(QIcon('resources/asset/pre_vis_1.png'))
        self.maxfev_alert.setIcon(QIcon('resources/asset/alert.png'))
        self.maxfev_alert.setToolTip('Optimal parameters not found: Number of calls to function has reached max fev = '
                                     '5000.')
        self.maxfev_alert.setVisible(JS08Settings.get('maxfev_flag'))

        self.consumer = Consumer(q)
        self.consumer.poped.connect(self.clock)
        self.consumer.start()

        self.video_thread = CurveThread(_q)
        self.video_thread.poped.connect(self.print_data)
        self.video_thread.start()

        self.click_style = 'border: 1px solid red;'

        # self.alert.clicked.connect(self.alert_test)

        self.c_vis_label.mousePressEvent = self.unit_convert
        self.p_vis_label.mousePressEvent = self.unit_convert

        self.label_1hour.mouseDoubleClickEvent = self.thumbnail_click1
        self.label_2hour.mouseDoubleClickEvent = self.thumbnail_click2
        self.label_3hour.mouseDoubleClickEvent = self.thumbnail_click3
        self.label_4hour.mouseDoubleClickEvent = self.thumbnail_click4
        self.label_5hour.mouseDoubleClickEvent = self.thumbnail_click5
        self.label_6hour.mouseDoubleClickEvent = self.thumbnail_click6

        self.log_view.clicked.connect(self.logview_btn_click)
        self.setting_button.clicked.connect(self.setting_btn_click)

        JS08Settings.restore_value('maxfev_count')
        # seongmin
        self.lstm_model = LSTM_model()
        self.coulmns = ["prev", "Day sin", "Day cos", "sun_altitude", "sun_azimuth", "sun_yes_no"]
        self.lstm_df = pd.DataFrame(columns = self.coulmns)

        self.show()

    def alert_test(self):
        self.alert.setIcon(QIcon('resources/asset/red.png'))
        try:
            strFormat = '%-20s%-10s\n'
            strOut = strFormat % ('Azimuth', 'Visibility (m)')
            for k, v in self.visibility.items():
                v = str(float(v))
                strOut += strFormat % (k, v)
        except AttributeError:
            strOut = 'It has not measured yet.'
            pass
        vis = QMessageBox()
        vis.setStyleSheet('color:rgb(0,0,0);')
        vis.about(self, '8-Way Visibility', f'{strOut}')

    def reset_StyleSheet(self):
        self.label_1hour.setStyleSheet('')
        self.label_2hour.setStyleSheet('')
        self.label_3hour.setStyleSheet('')
        self.label_4hour.setStyleSheet('')
        self.label_5hour.setStyleSheet('')
        self.label_6hour.setStyleSheet('')

    def thumbnail_view(self, file_name: str):
        self.view = ThumbnailView(file_name, int(file_name[2:8]))
        self.view.setGeometry(QRect(self.video_horizontalLayout.geometry().x(),
                                    self.video_horizontalLayout.geometry().y(),
                                    self.video_horizontalLayout.geometry().width(),
                                    self.video_horizontalLayout.geometry().height()))
        self.view.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.view.setWindowModality(Qt.ApplicationModal)
        self.view.show()
        self.view.raise_()

    def thumbnail_show(self):
        self.thumbnail_info_label.setStyleSheet('color: #1c88e3; background-color: #1b3146')
        self.thumbnail_info_label.setText('')
        self.reset_StyleSheet()
        self.view.close()

    def logview_btn_click(self):
        dlg = LogView()
        dlg.show()
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.exec()

    @Slot()
    def setting_btn_click(self):
        # self.front_video_widget.media_player.stop()
        # self.rear_video_widget.media_player.stop()
        # self.consumer.pause()

        # dlg = JS08SettingWidget()
        # dlg.show()
        # dlg.setWindowModality(Qt.ApplicationModal)
        # dlg.exec()

        # self.front_video_widget.media_player.play()
        # self.rear_video_widget.media_player.play()
        # self.consumer.resume()
        # self.consumer.start()

        if JS08Settings.get('right') == 'administrator':
            self.front_video_widget.media_player.stop()
            self.rear_video_widget.media_player.stop()
            self.consumer.pause()

            dlg = JS08AdminSettingWidget()
            dlg.show()
            dlg.setWindowModality(Qt.ApplicationModal)
            dlg.exec()

            self.front_video_widget.media_player.play()
            self.rear_video_widget.media_player.play()
            self.consumer.resume()
            self.consumer.start()

        elif JS08Settings.get('right') == 'user':
            self.front_video_widget.media_player.stop()
            self.rear_video_widget.media_player.stop()
            self.consumer.pause()

            dlg = JS08UserSettingWidget()
            dlg.show()
            dlg.setWindowModality(Qt.ApplicationModal)
            dlg.exec()

            self.front_video_widget.media_player.play()
            self.rear_video_widget.media_player.play()
            self.consumer.resume()
            self.consumer.start()

    def get_data(self, year, month_day):

        save_path = os.path.join(f'{JS08Settings.get("data_csv_path")}/Prevailing_Visibility/{year}')

        if os.path.isfile(f'{save_path}/{month_day}.csv'):
            self.result = pd.read_csv(f'{save_path}/{month_day}.csv')
            data_visibility = self.result['prev'].tolist()

            return data_visibility

        else:
            return []

    @Slot(str)
    def print_data(self, visibility: dict):
        """
        A function that runs every minute, updating data such as visibility values

        :param visibility: 8-degree Visibility value
        """

        self.convert_visibility(visibility)
        # visibility_front = visibility.get('visibility_front')
        # visibility_rear = visibility.get('visibility_rear')

        # graph visibility value
        self.graph_visibility_value.append(self.prevailing_visibility / 1000)
        if len(self.graph_visibility_value) >= 10:
            del self.graph_visibility_value[0]
        plot_value = round(float(np.mean(self.graph_visibility_value)), 3)

        l_epoch = QDateTime.currentSecsSinceEpoch()
        current_time = time.strftime('%Y-%m-%d %H:%M:00', time.localtime(l_epoch))
        epoch =  datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S").timestamp()
        _time = time.strftime('%Y%m%d%H%M%S', time.localtime(epoch))
        year = current_time[:4]
        md = current_time[5:7] + current_time[8:10]

        if _time[-4:] == '0000':
            self.front_video_widget.get_status()
            self.rear_video_widget.get_status()

        self.q_list = self.get_data(year, md)

        if len(self.q_list) == 0 or self.q_list_scale != len(self.q_list):
            self.q_list = []
            for i in range(self.q_list_scale):
                self.q_list.append(plot_value)
            result_vis = np.mean(self.q_list)
        else:
            self.q_list.pop(0)
            self.q_list.append(plot_value)
            result_vis = np.mean(self.q_list)

        if len(self.data_date) >= self.q_list_scale:
            self.data_date.pop(0)
            self.data_time.pop(0)

        self.data_date.append(current_time)
        self.data_time.append(epoch * 1000.0)

        save_path_front = os.path.join(
            f'{JS08Settings.get("data_csv_path")}/{JS08Settings.get("front_camera_name")}/{year}')
        save_path_rear = os.path.join(
            f'{JS08Settings.get("data_csv_path")}/{JS08Settings.get("rear_camera_name")}/{year}')
        save_path_prevailing = os.path.join(f'{JS08Settings.get("data_csv_path")}/Prevailing_Visibility/{year}')
        file_front = f'{save_path_front}/{md}.csv'
        file_rear = f'{save_path_rear}/{md}.csv'
        file_prevailing = f'{save_path_prevailing}/{md}.csv'

        result_front = pd.DataFrame(columns=['date', 'epoch', 'visibility', 'NE', 'EN', 'ES', 'SE'])
        result_rear = pd.DataFrame(columns=['date', 'epoch', 'visibility', 'SW', 'WS', 'WN', 'NW'])
        result_prevailing = pd.DataFrame(columns=['date', 'epoch', 'prev'])

        if os.path.isfile(f'{file_front}') is False or os.path.isfile(f'{file_rear}') is False \
                or os.path.isfile(f'{file_prevailing}') is False:
            os.makedirs(f'{save_path_front}', exist_ok=True)
            os.makedirs(f'{save_path_rear}', exist_ok=True)
            os.makedirs(f'{save_path_prevailing}', exist_ok=True)
            result_front.to_csv(f'{file_front}', mode='w', index=False)
            result_rear.to_csv(f'{file_rear}', mode='w', index=False)
            result_prevailing.to_csv(f'{file_prevailing}', mode='w', index=False)

        try:
            result_front['date'] = [self.data_date[-1]]
            result_front['epoch'] = [self.data_time[-1]]
            # result_front['visibility'] = visibility_front
            result_front['NE'] = visibility.get('NE')
            result_front['EN'] = visibility.get('EN')
            result_front['ES'] = visibility.get('ES')
            result_front['SE'] = visibility.get('SE')

            result_rear['date'] = [self.data_date[-1]]
            result_rear['epoch'] = [self.data_time[-1]]
            # result_rear['visibility'] = visibility_rear
            result_rear['SW'] = visibility.get('SW')
            result_rear['WS'] = visibility.get('WS')
            result_rear['WN'] = visibility.get('WN')
            result_rear['NW'] = visibility.get('NW')

            result_prevailing['date'] = [self.data_date[-1]]
            result_prevailing['epoch'] = [self.data_time[-1]]
            result_prevailing['prev'] = round(self.prevailing_visibility / 1000, 3)

        except TypeError as e:
            print(f'Occurred error ({current_time}) -\n{e}')

        result_rear.to_csv(f'{file_rear}', mode='a', index=False, header=False)
        result_front.to_csv(f'{file_front}', mode='a', index=False, header=False)
        result_prevailing.to_csv(f'{file_prevailing}', mode='a', index=False, header=False)

        # self.visibility_front = round(float(result_vis), 3)

        self._plot.refresh_stats(self.data_time[-1], self.q_list, self.predict_vis_meter)
        self._polar.refresh_stats(visibility)

        self.maxfev_alert.setVisible(JS08Settings.get('maxfev_flag'))

    @Slot(str)
    def clock(self, data):

        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(data)))
        self.year_date = current_time[2:4] + current_time[5:7] + current_time[8:10]
        self.real_time_label.setText(current_time)

        try:
            if self.km_mile_convert:
                self.c_vis_label.setText(f'{format(round(self.prevailing_visibility / 1609, 2), ",")} mile')
                self.p_vis_label.setText(f'{format(round(self.predict_vis_meter/1609, 2), ",")} mile')
            elif self.km_mile_convert is False:
                if self.visibility is not None:
                    self.c_vis_label.setText(
                        f'{format(int(self.prevailing_visibility), ",")} m')
                    if self.predict_vis_meter is not None:
                        self.p_vis_label.setText(f'{format(int(self.predict_vis_meter), ",")} m')
        except TypeError:
            pass

        if current_time[-2:] == '00':
            self.thumbnail_refresh()

            # if int(self.prevailing_visibility) <= JS08Settings.get('visibility_alert_limit'):
            #     self.alert.setIcon(QIcon('resources/asset/red.png'))
            # else:
            #     self.alert.setIcon(QIcon('resources/asset/green.png'))

    def convert_visibility(self, data: dict):
        """
        Function with airstrip visibility unit conversion algorithm applied.
        Currently, 'JS-08' stores visibility values in dict format. It can be converted at once.

        Notes
        --------
        - If visibility ranging from 0 m to less than 400 m, mark it in units of 25 m.
        - If visibility ranging from 400 m to less than 800 m, mark it in units of 50 m.
        .. math:: q (10 ^ { size - 1 } ) + re

        - If the visibility is more than 800 m, mark it in units of 25 m.
        .. math:: q (10 ^{ 2 } ) + re

        Examples
        --------
        >>> Visibility = {'A': 1263, 'B': 695, 'C': 341}
        >>> convert_visibility(Visibility)
        {'A': 1200, 'B': 650, 'C': 325}

        :param data: Visibility data in form of dict
        :return: Visibility data in Converted form of dict
        """

        keys = list(data.keys())
        values = []

        for i in keys:
            if type(data.get(i)) is bool:
                continue
            value = int(float(data.get(i)) * 1000)

            q, re = divmod(value, 100)
            size = len(str(value))

            if value < 400:
                if 0 <= re < 25:
                    re = 0
                elif 25 <= re < 50:
                    re = 25
                elif 50 <= re < 75:
                    re = 50
                elif 75 <= re < 100:
                    re = 75
                data[i] = (q * (10 ** (size - 1)) + re) / 1000

            elif 400 <= value < 800:
                if 0 <= re < 50:
                    re = 0
                elif 50 <= re < 100:
                    re = 50
                data[i] = (q * (10 ** (size - 1)) + re) / 1000

            elif 800 <= value:
                data[i] = (q * (10 ** 2)) / 1000

        self.visibility = data
        disposable = self.visibility.copy()
        # del disposable['visibility_front']
        # del disposable['visibility_rear']
        for i in disposable.keys():
            values.append(int(disposable.get(i) * 1000))

        values.sort(reverse=True)
        self.prevailing_visibility = values[3]

        if int(self.prevailing_visibility) <= JS08Settings.get('visibility_alert_limit'):
                self.alert.setIcon(QIcon('resources/asset/red.png'))
        else:
            self.alert.setIcon(QIcon('resources/asset/green.png'))
        
        # seongmin

        pv = self.prevailing_visibility /1000    

        front_target_name, rear_target_name = JS08Settings.get('front_camera_name'), JS08Settings.get('rear_camera_name')
        front_target_path = os.path.join(f'{JS08Settings.get("target_csv_path")}/{front_target_name}/{front_target_name}.csv')
        rear_target_path = os.path.join(f'{JS08Settings.get("target_csv_path")}/{rear_target_name}/{rear_target_name}.csv')
        ft, rt = pd.read_csv(front_target_path)['distance'].tolist(), pd.read_csv(rear_target_path)['distance'].tolist()
        distance_list = list(set(ft + rt))
        distance_list = sorted(distance_list)
        print("distance_list : ", distance_list)
        result = max(ft + rt)
        day = 24*60*60
        # 현재 시간 저장
        # now = pd.Timestamp.now()
        now_str = time.strftime('%Y-%m-%d %H:%M:00', time.localtime(time.time()))
        now = datetime.strptime(now_str, "%Y-%m-%d %H:%M:%S")
        current_time =  now.timestamp()
        print("current_time : ", current_time)
        
        # if self.lstm_df.shape[0] == 0 or now.minute % 10 == 0:
        #     pass
        # else:
        #     return
        
        ten_m_later = now + timedelta(minutes=10)
        predict_epoch_10m = ten_m_later.timestamp()
        
        ten_h_later = now + timedelta(hours=1)
        predict_epoch_1h = ten_h_later.timestamp()

        if self.lstm_df.shape[0] < 18:
            
            if self.lstm_df.shape[0] == 0:
                for i in range(18, 0, -1):
                    before_time = now - timedelta(minutes=10 * i)
                    # print("now : ", before_time)
                    sun_alt, sun_azm, sun_yn = self.calculate_sun_altitude(before_time)
                    new_serires = {'prev':  pv, "Day sin" : np.sin(before_time.timestamp() * (2 * np.pi / day)), "Day cos" : np.cos(before_time.timestamp() * (2 * np.pi / day)),
                                   "sun_altitude" : sun_alt,"sun_azimuth" : sun_azm,"sun_yes_no" : sun_yn}
                    self.lstm_df = self.lstm_df.append(new_serires, ignore_index = True)
                
            # if self.lstm_df.shape[0] == 0:
            #     before_time = now - timedelta(minutes=15)
            #     new_serires = {'prev':  pv, "Day sin" : np.sin(before_time.timestamp() * (2 * np.pi / day)), "Day cos" : np.cos(before_time.timestamp() * (2 * np.pi / day))}
            #     self.lstm_df = self.lstm_df.append(new_serires, ignore_index = True)

            new_serires = {'prev':  pv, "Day sin" : np.sin(current_time * (2 * np.pi / day)), "Day cos" : np.cos(current_time * (2 * np.pi / day)),
                           "sun_altitude" : sun_alt,"sun_azimuth" : sun_azm,"sun_yes_no" : sun_yn}
            self.lstm_df = self.lstm_df.append(new_serires, ignore_index = True)

            print("self.lstm_df", self.lstm_df.shape)
            # data_df = (self.lstm_df - df_mean) / df_std
            # print(data_df.head())
            
            prediction_cvt = self.lstm_model.predict_visibility(current_time, self.lstm_df)
            # 1시간 뒤 예측 값
            predict_vis_1h = prediction_cvt[0][5][0]
            predict_vis_1h = self.lstm_model.find_closest_value(predict_vis_1h, distance_list)
            
            self.predict_vis_meter = predict_vis_1h * 1000
            self._predic_plot.appendData(pv, prediction_cvt)
            
            # 10분 뒤 예측값
            predict_vis_10m = prediction_cvt[0][1][0]
            predict_vis_10m = self.lstm_model.find_closest_value(predict_vis_10m, distance_list)
            
            self.predict_vis_meter_10m = predict_vis_10m * 1000
            
            current_time_str = now.strftime('%Y%m%d%H%M00')
            predict_path = f"predict/{current_time_str[:8]}"
            predict_vis_file = current_time_str[:8] + ".csv"
            predict_file_path =  os.path.join(predict_path, predict_vis_file)
            print("predict__file__name : ", predict_vis_file)
            
            if os.path.isdir(f'{predict_path}') is False:
                os.makedirs(predict_path, exist_ok=True)
                df_predict = pd.DataFrame(columns=['date', 'epoch', 'predict_epoch_1h', 'predict_value_1h','predict_epoch_10m', 'predict_value_10m'])
                df_predict.to_csv(predict_file_path, mode='w', index=False)
                print("create predict path")
                
            else:
                df_predict = pd.read_csv(predict_file_path)
            
            df_predict = pd.concat([df_predict, pd.DataFrame([[now.strftime('%Y-%m-%d %H:%M:00'), (current_time*1000.0), (predict_epoch_1h*1000.0), str(int(predict_vis_1h)), (predict_epoch_10m*1000.0), str(int(predict_vis_10m))]],
                                                            columns=['date', 'epoch', 'predict_epoch_1h', 'predict_value_1h','predict_epoch_10m', 'predict_value_10m'])], join='outer')
            df_predict.to_csv(predict_file_path, mode='w', index=False)
            print("create predict file")
             
        elif now.minute % 10 == 0:
            print(self.lstm_df.head())
            self.lstm_df.drop([0], axis=0, inplace=True)
            sun_alt, sun_azm, sun_yn = self.calculate_sun_altitude(now)
            new_serires = {'prev':  pv, "Day sin" : np.sin(current_time * (2 * np.pi / day)), "Day cos" : np.cos(current_time * (2 * np.pi / day)),
                           "sun_altitude" : sun_alt,"sun_azimuth" : sun_azm,"sun_yes_no" : sun_yn}
            self.lstm_df = self.lstm_df.append(new_serires, ignore_index = True)
            # data_df = (self.lstm_df - df_mean) / df_std
            self.lstm_df = self.lstm_df.reset_index(drop=True)
            # print(data_df.head())


            prediction_cvt = self.lstm_model.predict_visibility(current_time, self.lstm_df)
            # 1시간 뒤 예측 값
            predict_vis_1h = prediction_cvt[0][5][0]
            predict_vis_1h = self.lstm_model.find_closest_value(predict_vis_1h, distance_list)
            
            self.predict_vis_meter = predict_vis_1h * 1000
            self._predic_plot.appendData(pv, prediction_cvt)
            
            # 10분 뒤 예측값
            predict_vis_10m = prediction_cvt[0][0][0]
            predict_vis_10m = self.lstm_model.find_closest_value(predict_vis_10m, distance_list)
            
            self.predict_vis_meter_10m = predict_vis_10m * 1000
            # self.predict_vis_meter = 0
            # self.predict_vis_meter_10m = 0
                



            # if self.predict_vis_meter > result * 1000:
            #     self.predict_vis_meter = result * 1000
            # else:
            #     pass
                
                
            current_time_str = now.strftime('%Y%m%d%H%M00')
            predict_path = f"predict/{current_time_str[:8]}"
            predict_vis_file = current_time_str[:8] + ".csv"
            predict_file_path =  os.path.join(predict_path, predict_vis_file)
            print("predict__file__name : ", predict_vis_file)
            
            if os.path.isdir(f'{predict_path}') is False:
                os.makedirs(predict_path, exist_ok=True)
                df_predict = pd.DataFrame(columns=['date', 'epoch', 'predict_epoch_1h', 'predict_value_1h','predict_epoch_10m', 'predict_value_10m'])
                df_predict.to_csv(predict_file_path, mode='w', index=False)
                print("create predict path")
                
            else:
                df_predict = pd.read_csv(predict_file_path)
            
            df_predict = pd.concat([df_predict, pd.DataFrame([[now.strftime('%Y-%m-%d %H:%M:00'), (current_time*1000.0), (predict_epoch_1h*1000.0), str(int(predict_vis_1h)), (predict_epoch_10m*1000.0), str(int(predict_vis_10m))]],
                                                            columns=['date', 'epoch', 'predict_epoch_1h', 'predict_value_1h','predict_epoch_10m', 'predict_value_10m'])], join='outer')
            df_predict.to_csv(predict_file_path, mode='w', index=False)
            print("create predict file")

    # 태양과 지평선 사이의 고도각, 방위각, 태양 유무 추가
    def calculate_sun_altitude(self, dt_value):
        observer = ephem.Observer()
        observer.lat = '37.564214'  # 위도 (서울의 위도)
        observer.lon = '127.001699'  # 경도 (서울의 경도)
        observer.date = dt_value - timedelta(hours=9)

        sun = ephem.Sun(observer)    
        altitude = sun.alt  # 태양의 고도각
        azimuth = sun.az  # 태양의 방위각
        
        if math.degrees(altitude) > 0:
            sun_yn = 1
        else:
            sun_yn = 0
            
        return math.degrees(altitude), math.degrees(azimuth), sun_yn


    def thumbnail_refresh(self):

        try:
            data_datetime = self.result['date'].tolist()
        except TypeError:
            data_datetime = []

        one_hour_ago = time.strftime('%Y%m%d%H%M00', time.localtime(time.time() - 3600))
        one_hour_visibility = time.strftime('%Y-%m-%d %H:%M:00', time.localtime(time.time() - 3600))
        two_hour_ago = time.strftime('%Y%m%d%H%M00', time.localtime(time.time() - 3600 * 2))
        two_hour_visibility = time.strftime('%Y-%m-%d %H:%M:00', time.localtime(time.time() - 3600 * 2))
        three_hour_ago = time.strftime('%Y%m%d%H%M00', time.localtime(time.time() - 3600 * 3))
        three_hour_visibility = time.strftime('%Y-%m-%d %H:%M:00', time.localtime(time.time() - 3600 * 3))
        four_hour_ago = time.strftime('%Y%m%d%H%M00', time.localtime(time.time() - 3600 * 4))
        four_hour_visibility = time.strftime('%Y-%m-%d %H:%M:00', time.localtime(time.time() - 3600 * 4))
        five_hour_ago = time.strftime('%Y%m%d%H%M00', time.localtime(time.time() - 3600 * 5))
        five_hour_visibility = time.strftime('%Y-%m-%d %H:%M:00', time.localtime(time.time() - 3600 * 5))
        six_hour_ago = time.strftime('%Y%m%d%H%M00', time.localtime(time.time() - 3600 * 6))
        six_hour_visibility = time.strftime('%Y-%m-%d %H:%M:00', time.localtime(time.time() - 3600 * 6))

        if one_hour_visibility in data_datetime:
            data = self.result.where(self.result['date'] == one_hour_visibility).dropna()
            vis = int(data['prev'].tolist()[0] * 1000)
            self.label_1hour_time.setText(f'{time.strftime("%H:%M", time.localtime(time.time() - 3600))}'
                                          f' - {vis} m')
        else:
            self.label_1hour_time.setText(f'{time.strftime("%H:%M", time.localtime(time.time() - 3600))}')

        if two_hour_visibility in data_datetime:
            data = self.result.where(self.result['date'] == two_hour_visibility).dropna()
            vis = int(data['prev'].tolist()[0] * 1000)
            self.label_2hour_time.setText(f'{time.strftime("%H:%M", time.localtime(time.time() - 3600 * 2))}'
                                          f' - {vis} m')
        else:
            self.label_2hour_time.setText(f'{time.strftime("%H:%M", time.localtime(time.time() - 3600 * 2))}')

        if three_hour_visibility in data_datetime:
            data = self.result.where(self.result['date'] == three_hour_visibility).dropna()
            vis = int(data['prev'].tolist()[0] * 1000)
            self.label_3hour_time.setText(f'{time.strftime("%H:%M", time.localtime(time.time() - 3600 * 3))}'
                                          f' - {vis} m')
        else:
            self.label_3hour_time.setText(f'{time.strftime("%H:%M", time.localtime(time.time() - 3600 * 3))}')

        if four_hour_visibility in data_datetime:
            data = self.result.where(self.result['date'] == four_hour_visibility).dropna()
            vis = int(data['prev'].tolist()[0] * 1000)
            self.label_4hour_time.setText(f'{time.strftime("%H:%M", time.localtime(time.time() - 3600 * 4))}'
                                          f' - {vis} m')
        else:
            self.label_4hour_time.setText(f'{time.strftime("%H:%M", time.localtime(time.time() - 3600 * 4))}')

        if five_hour_visibility in data_datetime:
            data = self.result.where(self.result['date'] == five_hour_visibility).dropna()
            vis = int(data['prev'].tolist()[0] * 1000)
            self.label_5hour_time.setText(f'{time.strftime("%H:%M", time.localtime(time.time() - 3600 * 5))}'
                                          f' - {vis} m')
        else:
            self.label_5hour_time.setText(f'{time.strftime("%H:%M", time.localtime(time.time() - 3600 * 5))}')

        if six_hour_visibility in data_datetime:
            data = self.result.where(self.result['date'] == six_hour_visibility).dropna()
            vis = int(data['prev'].tolist()[0] * 1000)
            self.label_6hour_time.setText(f'{time.strftime("%H:%M", time.localtime(time.time() - 3600 * 6))}'
                                          f' - {vis} m')
        else:
            self.label_6hour_time.setText(f'{time.strftime("%H:%M", time.localtime(time.time() - 3600 * 6))}')

        if os.path.isfile(
                f'{JS08Settings.get("image_save_path")}/thumbnail/'
                f'{JS08Settings.get("front_camera_name")}/{self.year_date}/{one_hour_ago}.jpg'):
            self.label_1hour.setPixmap(
                QPixmap(f'{JS08Settings.get("image_save_path")}/thumbnail/'
                        f'{JS08Settings.get("front_camera_name")}/{self.year_date}/{one_hour_ago}.jpg')
                    .scaled(self.label_1hour.width(), self.label_1hour.height(), Qt.IgnoreAspectRatio))

        if os.path.isfile(
                f'{JS08Settings.get("image_save_path")}/thumbnail/'
                f'{JS08Settings.get("front_camera_name")}/{self.year_date}/{two_hour_ago}.jpg'):
            self.label_2hour.setPixmap(
                QPixmap(f'{JS08Settings.get("image_save_path")}/thumbnail/'
                        f'{JS08Settings.get("front_camera_name")}/{self.year_date}/{two_hour_ago}.jpg')
                    .scaled(self.label_2hour.width(), self.label_2hour.height(), Qt.IgnoreAspectRatio))

        if os.path.isfile(
                f'{JS08Settings.get("image_save_path")}/thumbnail/'
                f'{JS08Settings.get("front_camera_name")}/{self.year_date}/{three_hour_ago}.jpg'):
            self.label_3hour.setPixmap(
                QPixmap(f'{JS08Settings.get("image_save_path")}/thumbnail/'
                        f'{JS08Settings.get("front_camera_name")}/{self.year_date}/{three_hour_ago}.jpg')
                    .scaled(self.label_3hour.width(), self.label_3hour.height(), Qt.IgnoreAspectRatio))

        if os.path.isfile(
                f'{JS08Settings.get("image_save_path")}/thumbnail/'
                f'{JS08Settings.get("front_camera_name")}/{self.year_date}/{four_hour_ago}.jpg'):
            self.label_4hour.setPixmap(
                QPixmap(f'{JS08Settings.get("image_save_path")}/thumbnail/'
                        f'{JS08Settings.get("front_camera_name")}/{self.year_date}/{four_hour_ago}.jpg')
                    .scaled(self.label_4hour.width(), self.label_4hour.height(), Qt.IgnoreAspectRatio))

        if os.path.isfile(
                f'{JS08Settings.get("image_save_path")}/thumbnail/'
                f'{JS08Settings.get("front_camera_name")}/{self.year_date}/{five_hour_ago}.jpg'):
            self.label_5hour.setPixmap(
                QPixmap(f'{JS08Settings.get("image_save_path")}/thumbnail/'
                        f'{JS08Settings.get("front_camera_name")}/{self.year_date}/{five_hour_ago}.jpg')
                    .scaled(self.label_5hour.width(), self.label_5hour.height(), Qt.IgnoreAspectRatio))

        if os.path.isfile(
                f'{JS08Settings.get("image_save_path")}/thumbnail/'
                f'{JS08Settings.get("front_camera_name")}/{self.year_date}/{six_hour_ago}.jpg'):
            self.label_6hour.setPixmap(
                QPixmap(f'{JS08Settings.get("image_save_path")}/thumbnail/'
                        f'{JS08Settings.get("front_camera_name")}/{self.year_date}/{six_hour_ago}.jpg')
                    .scaled(self.label_6hour.width(), self.label_6hour.height(), Qt.IgnoreAspectRatio))
        self.update()

    # Event
    def thumbnail_click1(self, e):
        name = self.label_1hour_time.text()[:2] + self.label_1hour_time.text()[3:5]
        epoch = time.strftime('%Y%m%d', time.localtime(time.time()))
        self.thumbnail_view(f'{epoch}{name}00')

        self.reset_StyleSheet()
        self.label_1hour.setStyleSheet(self.click_style)
        self.thumbnail_info_label.setStyleSheet('color: #ffffff; background-color: #1b3146')
        self.thumbnail_info_label.setText(f' {self.label_1hour_time.text()} image')

        QTimer.singleShot(5000, self.thumbnail_show)

    def thumbnail_click2(self, e):
        name = self.label_2hour_time.text()[:2] + self.label_2hour_time.text()[3:5]
        epoch = time.strftime('%Y%m%d', time.localtime(time.time()))
        self.thumbnail_view(f'{epoch}{name}00')

        self.reset_StyleSheet()
        self.label_2hour.setStyleSheet(self.click_style)
        self.thumbnail_info_label.setStyleSheet('color: #ffffff; background-color: #1b3146')
        self.thumbnail_info_label.setText(f' {self.label_2hour_time.text()} image')

        QTimer.singleShot(5000, self.thumbnail_show)

    def thumbnail_click3(self, e):
        name = self.label_3hour_time.text()[:2] + self.label_3hour_time.text()[3:5]
        epoch = time.strftime('%Y%m%d', time.localtime(time.time()))
        self.thumbnail_view(f'{epoch}{name}00')

        self.reset_StyleSheet()
        self.label_3hour.setStyleSheet(self.click_style)
        self.thumbnail_info_label.setStyleSheet('color: #ffffff; background-color: #1b3146')
        self.thumbnail_info_label.setText(f' {self.label_3hour_time.text()} image')

        QTimer.singleShot(5000, self.thumbnail_show)

    def thumbnail_click4(self, e):
        name = self.label_4hour_time.text()[:2] + self.label_4hour_time.text()[3:5]
        epoch = time.strftime('%Y%m%d', time.localtime(time.time()))
        self.thumbnail_view(f'{epoch}{name}00')

        self.reset_StyleSheet()
        self.label_4hour.setStyleSheet(self.click_style)
        self.thumbnail_info_label.setStyleSheet('color: #ffffff; background-color: #1b3146')
        self.thumbnail_info_label.setText(f' {self.label_4hour_time.text()} image')

        QTimer.singleShot(5000, self.thumbnail_show)

    def thumbnail_click5(self, e):
        name = self.label_5hour_time.text()[:2] + self.label_5hour_time.text()[3:5]
        epoch = time.strftime('%Y%m%d', time.localtime(time.time()))
        self.thumbnail_view(f'{epoch}{name}00')

        self.reset_StyleSheet()
        self.label_5hour.setStyleSheet(self.click_style)
        self.thumbnail_info_label.setStyleSheet('color: #ffffff; background-color: #1b3146')
        self.thumbnail_info_label.setText(f' {self.label_5hour_time.text()} image')

        QTimer.singleShot(5000, self.thumbnail_show)

    def thumbnail_click6(self, e):
        name = self.label_6hour_time.text()[:2] + self.label_6hour_time.text()[3:5]
        epoch = time.strftime('%Y%m%d', time.localtime(time.time()))
        self.thumbnail_view(f'{epoch}{name}00')

        self.reset_StyleSheet()
        self.label_6hour.setStyleSheet(self.click_style)
        self.thumbnail_info_label.setStyleSheet('color: #ffffff; background-color: #1b3146')
        self.thumbnail_info_label.setText(f' {self.label_6hour_time.text()} image')

        QTimer.singleShot(5000, self.thumbnail_show)

    def btn_on(self, event):
        self.setting_button.setIcon(QIcon('resources/asset/settings_on.png'))

    def btn_off(self, event):
        self.setting_button.setIcon(QIcon('resources/asset/settings.png'))

    def unit_convert(self, event):
        if self.km_mile_convert:
            self.km_mile_convert = False
        elif self.km_mile_convert is False:
            self.km_mile_convert = True

    def keyPressEvent(self, e):
        """Override function QMainwindow KeyPressEvent that works when key is pressed"""
        if e.key() == Qt.Key_F:
            self.showFullScreen()
            self.thumbnail_refresh()
        if e.key() == Qt.Key_D:
            self.showNormal()
            self.thumbnail_refresh()
        if e.modifiers() & Qt.ControlModifier:
            if e.key() == Qt.Key_W:
                self.close()
                sys.exit()

    def front_label_paintEvent(self, event):
        painter = QPainter(self.front_label)
        painter.setPen(QPen(Qt.white, 1, Qt.DotLine))

        painter.drawLine((self.front_label.width() * 0.25), 0,
                         (self.front_label.width() * 0.25), self.front_label.height())
        painter.drawLine((self.front_label.width() * 0.5), 0,
                         (self.front_label.width() * 0.5), self.front_label.height())
        painter.drawLine((self.front_label.width() * 0.75), 0,
                         (self.front_label.width() * 0.75), self.front_label.height())
        painter.drawLine((self.front_label.width() - 1), 0,
                         (self.front_label.width() - 1), self.front_label.height())

        painter.drawText(self.front_label.width() * 0.125, 14, 'NE')
        painter.drawText(self.front_label.width() * 0.375, 14, 'EN')
        painter.drawText(self.front_label.width() * 0.625, 14, 'ES')
        painter.drawText(self.front_label.width() * 0.875, 14, 'SE')

        painter.end()

    def rear_label_paintEvent(self, event):
        painter = QPainter(self.rear_label)
        painter.setPen(QPen(Qt.white, 1, Qt.DotLine))

        painter.drawLine((self.rear_label.width() * 0.25), 0,
                         (self.rear_label.width() * 0.25), self.rear_label.height())
        painter.drawLine((self.rear_label.width() * 0.5), 0,
                         (self.rear_label.width() * 0.5), self.rear_label.height())
        painter.drawLine((self.rear_label.width() * 0.75), 0,
                         (self.rear_label.width() * 0.75), self.rear_label.height())
        painter.drawLine((self.rear_label.width() - 1), 0,
                         (self.rear_label.width() - 1), self.rear_label.height())

        painter.drawText(self.rear_label.width() * 0.125, 14, 'SW')
        painter.drawText(self.rear_label.width() * 0.375, 14, 'WS')
        painter.drawText(self.rear_label.width() * 0.625, 14, 'WN')
        painter.drawText(self.rear_label.width() * 0.875, 14, 'NW')

        painter.end()

    def closeEvent(self, e):
        self.consumer.stop()
        self.video_thread.stop()
        print(f'Close time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        log(JS08Settings.get('current_id'), 'Logout, Program exit')


class VideoWidget(QWidget):
    """Video stream player using QVideoWidget"""

    def __init__(self, parent: QObject = None):
        super().__init__(parent)

        args = [
            '--rtsp-frame-buffer-size=400000',
            '--quiet',
            '--sout-x264-keyint=25',
        ]

        self.instance = vlc.Instance(args)
        self.instance.log_unset()

        self.media_player = self.instance.media_player_new()
        self.uri = None

        # Current camera must be 'PNM-9031RV'
        self.media_player.video_set_aspect_ratio('2:1')

        self.video_frame = QFrame()

        if sys.platform == 'win32':
            self.media_player.set_hwnd(self.video_frame.winId())

    def on_camera_change(self, uri: str):
        if uri[:4] == 'rtsp':
            self.uri = uri
            self.media_player.set_media(self.instance.media_new(self.uri))
            self.media_player.play()
        else:
            pass

    def get_status(self):
        # if self.media_player.is_playing() == 0:
        #     print(f'Player is not playing! in'
        #           f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(QDateTime.currentSecsSinceEpoch()))}')
        #     self.media_player.set_media(self.instance.media_new(self.uri))
        #     self.media_player.play()
        self.media_player.set_pause(1)
        self.media_player.play()


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QGuiApplication
    from model import JS08Settings

    mp.freeze_support()
    q = Queue()
    _q = Queue()

    _producer = producer

    p = Process(name='clock', target=clock_clock, args=(q,), daemon=True)
    _p = Process(name='producer', target=_producer, args=(_q,), daemon=True)

    p.start()
    # _p.start()

    os.makedirs(f'{JS08Settings.get("data_csv_path")}', exist_ok=True)
    os.makedirs(f'{JS08Settings.get("target_csv_path")}', exist_ok=True)
    os.makedirs(f'{JS08Settings.get("rgb_csv_path")}', exist_ok=True)
    os.makedirs(f'{JS08Settings.get("image_save_path")}', exist_ok=True)

    app = QApplication(sys.argv)
    screen_size = QGuiApplication.screens()[0].geometry()
    width, height = screen_size.width(), screen_size.height()
    if width > 1920 or height > 1080:
        QMessageBox.warning(None, 'Warning', 'JS08 is based on FHD screen.')
    window = JS08MainWindow(q, _q)
    sys.exit(app.exec())
