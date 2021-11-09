#!/usr/bin/env python3
#
# Copyright 2020-2021 Sijung Co., Ltd.
# 
# Authors: 
#     ruddyscent@gmail.com (Kyungwon Chun)
#     5jx2oh@gmail.com (Jongjin Oh)

import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

from PyQt5.QtCore import (QDateTime, QDir, QObject, QRect, QThread,
                          QThreadPool, QTime, QTimer, pyqtSignal, pyqtSlot)
from PyQt5.QtGui import QImage
from PyQt5.QtMultimedia import QVideoFrame

from .model import (Js06AttrModel, Js06CameraTableModel, Js06IoRunner,
                    Js06Settings, Js06SimpleTarget, Js06Wedge)


class Js06MainCtrl(QObject):
    abnormal_shutdown = pyqtSignal()
    front_camera_changed = pyqtSignal(str) # uri
    rear_camera_changed = pyqtSignal(str) # uri
    front_target_decomposed = pyqtSignal()
    rear_target_decomposed = pyqtSignal()
    target_assorted = pyqtSignal(list, list) # positives, negatives
    wedge_vis_ready = pyqtSignal(int, dict) # epoch, wedge visibility

    def __init__(self, model: Js06AttrModel):
        super().__init__()

        self.writer_pool = QThreadPool.globalInstance()
        self.writer_pool.setMaxThreadCount(1)

        self._model = model

        self.num_working_cam = 0

        self.front_decomposed_targets = []
        self.rear_decomposed_targets = []

        self.front_target_prepared = False
        self.rear_target_prepared = False

        self.init_db()

        self.observation_timer = QTimer(self)
        self.front_camera_changed.connect(self.decompose_front_targets)
        self.rear_camera_changed.connect(self.decompose_rear_targets)

        self.broker_running = False
        self.start_observation_timer()

    def init_db(self):
        db_host = Js06Settings.get('db_host')
        db_port = Js06Settings.get('db_port')
        db_name = Js06Settings.get('db_name')
        self._model.connect_to_db(db_host, db_port, db_name)

        if getattr(sys, 'frozen', False):
            directory = sys._MEIPASS
        else:
            directory = os.path.dirname(__file__)
        attr_path = os.path.join(directory, 'resources', 'attr.json')
        with open(attr_path, 'r') as f:
            attr_json = json.load(f)
        camera_path = os.path.join(directory, 'resources', 'camera.json')
        with open(camera_path, 'r') as f:
            camera_json = json.load(f)

        self._model.setup_db(attr_json, camera_json)

    @pyqtSlot(str)
    def decompose_front_targets(self, _: str) -> None:
        """Make list of SimpleTarget by decoposing compound targets.

        Parameters:
        """
        self.front_target_prepared = False
        self.decompose_targets('front')
        self.front_target_prepared = True

    @pyqtSlot(str)
    def decompose_rear_targets(self, _: str) -> None:
        """Make list of SimpleTarget by decoposing compound targets.

        Parameters:
        """
        self.rear_target_prepared = False
        self.decompose_targets('rear')
        self.rear_target_prepared = True

    def decompose_targets(self, direction: str) -> None:
        """Make list of SimpleTarget by decoposing compound targets.

        Parameters:
            direction: 'front' or 'rear', default is 'front'
        """
        decomposed_targets = []
        attr = self._model.read_attr()
        if direction == 'front':
            targets = attr['front_camera']['targets']
            id = str(attr['front_camera']['camera_id'])
        elif direction == 'rear':
            targets = attr['rear_camera']['targets']
            id = str(attr['rear_camera']['camera_id'])
        
        base_path = Js06Settings.get('image_base_path') 
        
        for tg in targets:
            wedge = tg['wedge']
            azimuth = tg['azimuth']
            point = tg['roi']['point']
            size = tg['roi']['size']
            roi = QRect(*point, *size)

            for i in range(len(tg['mask'])):
                label = f"{tg['label']}_{i}"
                distance = tg['distance'][i]
                mask_path = os.path.join(base_path, 'mask', id, tg['mask'][i])
                mask = self.read_mask(mask_path)
                st = Js06SimpleTarget(label, wedge, azimuth, distance, roi, mask)
                decomposed_targets.append(st)

        if direction == 'front':
            self.front_decomposed_targets = decomposed_targets
        elif direction == 'rear':
            self.rear_decomposed_targets = decomposed_targets

    def read_mask(self, path: str) -> np.ndarray:
        """Read mask image and return 

        Parameters:
            path: path to mask file
        """
        with open(path, 'rb') as f:
            content = f.read()
        image = QImage()
        image.loadFromData(content)
        return image

    def start_observation_timer(self) -> None:
        print('DEBUG(start_observation_timer):', QTime.currentTime().toString())
        self.observation_timer.setInterval(1000) # every one second
        self.observation_timer.timeout.connect(self.start_worker)
        self.observation_timer.start()

    @pyqtSlot()
    def start_worker(self) -> None:
        # if decomposed targets are not ready, quit.
        if self.front_target_prepared is False or self.rear_target_prepared is False:
            return

        # If broker is already running, quit.
        if self.broker_running:
            return
        else:
            self.broker_running = True

        self.epoch = QDateTime.currentSecsSinceEpoch()
        front_uri = self.get_front_camera_uri()
        rear_uri = self.get_rear_camera_uri()
        self.broker = Js06InferenceWorker(
            self.epoch,
            front_uri, 
            rear_uri, 
            self.front_decomposed_targets, 
            self.rear_decomposed_targets
            )
        self.broker_thread = QThread()
        self.broker.moveToThread(self.broker_thread)
        self.broker_thread.started.connect(self.broker.run)
        self.broker.finished.connect(self.broker_thread.quit)
        self.broker.finished.connect(self.postprocessing)
        self.broker.finished.connect(self.finalize_broker)
        self.broker_thread.start()

    @pyqtSlot()
    def finalize_broker(self):
        self.broker_running = False

    @pyqtSlot()
    def postprocessing(self):
        """
        epoch: seconds since epoch
        """
        epoch = self.epoch
            
        pos, neg = self.assort_discernment()
        self.target_assorted.emit(pos, neg)
        wedge_vis = self.wedge_visibility()
        self.wedge_vis_ready.emit(epoch, wedge_vis)
        self.write_visibilitiy(epoch, wedge_vis)

    @pyqtSlot()
    def stop_timer(self) -> None:
        self.observation_timer.stop()

    def assort_discernment(self) -> tuple:
        """Assort targets in positive or negative according to the discernment result
        """
        pos, neg = [], []

        for t in self.front_decomposed_targets:
            point = (t.azimuth, t.distance)
            if t.discernment:
                pos.append(point)
            else:
                neg.append(point)

        for t in self.rear_decomposed_targets:
            point = (t.azimuth, t.distance)
            if t.discernment:
                pos.append(point)
            else:
                neg.append(point)

        return pos, neg

    def write_visibilitiy(self, epoch: int, wedge_visibility: dict) -> None:
        wedge_visibility = wedge_visibility.copy()
        vis_list = list(wedge_visibility.values())
        prevailing = self.prevailing_visibility(vis_list)
        wedge_visibility['epoch'] = epoch
        wedge_visibility['prevailing'] = prevailing
        print('DEBUG:', wedge_visibility)
        self._model.write_visibility(wedge_visibility)

    def wedge_visibility(self) -> dict:
        wedge_vis = {w: None for w in Js06Wedge}
        for t in self.front_decomposed_targets:
            if t.discernment:
                if wedge_vis[t.wedge] == None:
                    wedge_vis[t.wedge] = t.distance
                elif wedge_vis[t.wedge] < t.distance:
                    wedge_vis[t.wedge] = t.distance
        for t in self.rear_decomposed_targets:
            if t.discernment:
                if wedge_vis[t.wedge] == None:
                    wedge_vis[t.wedge] = t.distance
                elif wedge_vis[t.wedge] < t.distance:
                    wedge_vis[t.wedge] = t.distance
        return wedge_vis

    def prevailing_visibility(self, wedge_vis: list) -> float:
        if None in wedge_vis:
            return None
        sorted_vis = sorted(wedge_vis, reverse=True)
        prevailing = sorted_vis[(len(sorted_vis) - 1) // 2]
        return prevailing

    def save_image(self, dir: str, filename: str, image: QImage) -> None:
        os.makedirs(dir, exist_ok=True)
        path = QDir.cleanPath(os.path.join(dir, filename))
        runner = Js06IoRunner(path, image)
        self.writer_pool.start(runner)
    
    def grab_image(self, direction: str) -> QImage:
        """
        Parameters:
            direction: 'front' or 'rear'
        """
        if direction == 'front':
            uri = self.get_front_camera_uri()
        elif direction == 'rear':
            uri = self.get_rear_camera_uri()
        cap = cv2.VideoCapture(uri)
        ret, frame = cap.read()
        image = None
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        return image

    @pyqtSlot(QVideoFrame)
    def update_front_video_frame(self, video_frame: QVideoFrame) -> None:
        self.front_video_frame = video_frame

    @pyqtSlot(QVideoFrame)
    def update_rear_video_frame(self, video_frame: QVideoFrame) -> None:
        self.rear_video_frame = video_frame

    @pyqtSlot()
    def get_front_camera_uri(self) -> str:
        attr = self._model.read_attr()
        return attr['front_camera']['uri']

    @pyqtSlot()
    def get_rear_camera_uri(self) -> str:
        attr = self._model.read_attr()
        return attr['rear_camera']['uri']

    def get_front_target(self) -> list:
        attr = self._model.read_attr()
        return attr['front_camera']['targets']

    def get_rear_target(self) -> list:
        attr = self._model.read_attr()
        return attr['rear_camera']['targets']

    def get_camera_table_model(self) -> dict:
        cameras = self.get_cameras()
        table_model =  Js06CameraTableModel(cameras)
        return table_model

    def check_exit_status(self) -> bool:
        normal_exit = Js06Settings.get('normal_shutdown')
        Js06Settings.set('normal_shutdown', False)
        return normal_exit

    def update_cameras(self, cameras: list, update_target: bool = False) -> None:
        # Remove deleted cameras
        cam_id_in_db = [cam["_id"] for cam in self._model.read_cameras()]
        cam_id_in_arg = [cam["_id"] for cam in cameras]
        for cam_id in cam_id_in_db:
            if cam_id not in cam_id_in_arg:
                self._model.delete_camera(cam_id)
    
        # if `cameras` does not have 'targets' field, add an empty list for it.
        for cam in cameras:
            if 'targets' not in cam:
                cam['targets'] = []

        # Copy targets if `update_target` is False.
        if update_target == False:
            cam_in_db = self._model.read_cameras()
            for c_db in cam_in_db:
                for c_arg in cameras:
                    if c_arg['_id'] == c_db['_id']:
                        c_arg['targets'] = c_db['targets']
                        continue
        
        # if '_id' is empty, delete the field
        for cam in cameras:
            if not cam['_id']:
                del cam['_id']

        # Update existing camera or Insert new cameras
        for cam in cameras:
            self._model.upsert_camera(cam)

    @pyqtSlot()
    def close_process(self) -> None:
        Js06Settings.set('normal_shutdown', True)

    def get_attr(self) -> dict:
        attr_doc = self._model.read_attr()
        # attr_doc = None
        # if self._attr.count_documents({}):
        #     attr_doc = list(self._attr.find().sort("_id", -1).limit(1))[0]
        return attr_doc
    
    def insert_attr(self, model: dict) -> None:
        self._model.insert_attr(model)

    @pyqtSlot()
    def restore_defaults(self) -> None:
        Js06Settings.restore_defaults()

    @pyqtSlot(bool)
    def set_normal_shutdown(self) -> None:
         Js06Settings.set('normal_shutdown', True)

    def get_cameras(self) -> list:
        return self._model.read_cameras()


class Js06InferenceWorker(QObject):
    finished = pyqtSignal()
    
    def __init__(self, epoch: int, front_uri: str, rear_uri: str, front_decomposed_targets: list, rear_decomposed_targets: list) -> None:
        """
        Parameters:
            ctrl:
        """
        super().__init__()
        
        # TODO(Kyungwon): Put the model file into Qt Resource Collection.
        if getattr(sys, 'frozen', False):
            directory = sys._MEIPASS
        else:
            directory = os.path.dirname(__file__)

        self.epoch = epoch
        self.front_uri = front_uri
        self.rear_uri = rear_uri
        self.front_targets = front_decomposed_targets
        self.rear_targets = rear_decomposed_targets

        # num_threads = Js06Settings.get('inferece_thread_count')
        self.batch_size = Js06Settings.get('inference_batch_size')

        # Prepare model.
        model_path = os.path.join(directory, 'resources', 'js02.onnx')
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Prepare mask array.
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

    def grab_image(self, uri: str) -> QImage:
        """
        Parameters:
            uri: URI of a video stream
        """
        cap = cv2.VideoCapture(uri)
        ret, frame = cap.read()
        image = None
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        return image

    def save_image(self, dir: str, filename: str, image: QImage) -> None:
        os.makedirs(dir, exist_ok=True)
        path = QDir.cleanPath(os.path.join(dir, filename))
        image.save(path)

    def classify_image(self, image: np.ndarray) -> bool:
        """Discriminate the image.

        Return True if the model can discriminate the image.
        """
        input_data = image
        input_name = self.session.get_inputs()[0].name
        output_data = self.session.run(None, {input_name: input_data})
        results = np.squeeze(output_data)
        top = np.argmax(results)
        return top == 1

    def run(self):
        front_image = self.grab_image(self.front_uri)
        rear_image = self.grab_image(self.rear_uri)

        if front_image is None:
            print('DEBUG: Failed to capture the front video stream')
            self.finished.emit()
            return

        if rear_image is None:
            print('DEBUG: Failed to capture the rear video stream')
            self.finished.emit()
            return

        if Js06Settings.get('save_vista'):
            basepath = Js06Settings.get('image_base_path')
            now = QDateTime.fromSecsSinceEpoch(self.epoch)
            dir = os.path.join(basepath, 'vista', now.toString("yyyy-MM-dd"))
            filename = f'vista-front-{now.toString("yyyy-MM-dd-hh-mm")}.png'
            self.save_image(dir, filename, front_image)
            filename = f'vista-rear-{now.toString("yyyy-MM-dd-hh-mm")}.png'
            self.save_image(dir, filename, rear_image)
        
        # padding_size = self.batch_size - (len(self.front_targets) % self.batch_size)
        # result = np.zeros(len(self.front_targets) + padding_size)
        # for i, target in enumerate(self.front_targets):
        #     if i % self.batch_size == 0:
        #         data = np.zeros(
        #             (self.batch_size, self.input_height, self.input_width, 3), 
        #             dtype=np.float32
        #             )

        #     roi_image = target.clip_roi(front_image)
        #     arr = target.img_to_arr(roi_image, self.input_width, self.input_height)
        #     masked_arr = arr * target.mask
        #     data[i % self.batch_size] = masked_arr

        #     if i % self.batch_size == self.batch_size - 1:
        #         result[i: i + self.batch_size] = self.classify_image(data)

        # for i, target in enumerate(self.front_targets):
        #     target.discernment = result[i] is True

        # padding_size = self.batch_size - (len(self.rear_targets) % self.batch_size)
        # result = np.zeros(len(self.rear_targets) + padding_size)
        # for i, target in enumerate(self.rear_targets):
        #     if i % self.batch_size == 0:
        #         data = np.zeros(
        #             (self.batch_size, self.input_height, self.input_width, 3),
        #             np.float32
        #             )

        #     roi_image = target.clip_roi(rear_image)
        #     arr = target.img_to_arr(roi_image, self.input_width, self.input_height)
        #     masked_arr = arr * target.mask
        #     data[i % self.batch_size] = masked_arr

        #     if i % self.batch_size == self.batch_size - 1:
        #         result[i: i + self.batch_size] = self.classify_image(data)

        # for i, target in enumerate(self.rear_targets):
        #     target.discernment = result[i] is True

        for i, target in enumerate(self.front_targets):
            roi_image = target.clip_roi(front_image)
            arr = target.img_to_arr(roi_image, self.input_width, self.input_height)
            masked_arr = np.expand_dims(arr * target.mask, 0)
            result = self.classify_image(masked_arr)
            target.discernment = np.squeeze(result == 1)

        for i, target in enumerate(self.rear_targets):
            roi_image = target.clip_roi(rear_image)
            arr = target.img_to_arr(roi_image, self.input_width, self.input_height)
            masked_arr = np.expand_dims(arr * target.mask, 0)
            result = self.classify_image(masked_arr)
            target.discernment = np.squeeze(result == 1)

        self.finished.emit()
