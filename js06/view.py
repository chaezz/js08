#!/usr/bin/env python3
#
# Copyright 2020-2021 Sijung Co., Ltd.
# 
# Authors: 
#     ruddyscent@gmail.com (Kyungwon Chun)
#     5jx2oh@gmail.com (Jongjin Oh)

# # Set up backends of matplotlib
# from matplotlib.backends.qt_compat import QtCore
# if QtCore.qVersion() >= "5.":
#     from matplotlib.backends.backend_qt5agg import (
#         FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
# else:
#     from matplotlib.backends.backend_qt4agg import (
#         FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

import os

from PyQt5.QtCore import QObject, QUrl, Qt, pyqtSignal, pyqtSlot, QPersistentModelIndex
from PyQt5.QtGui import QCloseEvent, QPen, QMouseEvent, QMoveEvent, QPixmap, QImage, QPainter, QTransform
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer, QVideoFrame, QVideoProbe
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt5.QtWidgets import QDialog, QGraphicsRectItem, QGraphicsScene, \
    QGraphicsView, QMainWindow, QDockWidget, QMessageBox, QInputDialog, QVBoxLayout, \
    QWidget, QLabel
from PyQt5 import uic

from js06.controller import Js06MainCtrl

class Js06CameraView(QDialog):
    def __init__(self, controller: Js06MainCtrl):
        super().__init__()

        ui_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "../resources/camera_view.ui")
        uic.loadUi(ui_path, self)

        self._ctrl = controller
        self._model = self._ctrl.get_camera_table_model()
        self.tableView.setModel(self._model)
        self.insertAbove.clicked.connect(self.insert_above)
        self.insertBelow.clicked.connect(self.insert_below)
        self.removeRows.clicked.connect(self.remove_rows)
        self.buttonBox.accepted.connect(self.save_cameras)
    # end of __init__

    def insert_above(self):
        selected = self.tableView.selectedIndexes()
        row = selected[0].row() if selected else 0
        self._model.insertRows(row, 1, None)
    # end of insert_above

    def insert_below(self):
        selected = self.tableView.selectedIndexes()
        row = selected[-1].row() if selected else self._model.rowCount(None)
        self._model.insertRows(row + 1, 1, None)
    # end of insert_below

    def remove_rows(self):
        selected = self.tableView.selectedIndexes()
        if selected:
            self._model.removeRows(selected[0].row(), len(selected), None)
    # end of remove_rows

    def save_cameras(self):
        cameras = self._model.get_data()
        self._ctrl.update_cameras(cameras)
    # end of save_cameras

    def accepted(self):
        index = self.tableView.currentIndex()
        NewIndex = self.tableView.model().index(index.row(), 6)
        add = NewIndex.data()
        print(f"Select uri: [{add}]")
        index_list = []
        for model_index in self.tableView.selectionModel().selectedRows():
            index = QPersistentModelIndex(model_index)
            index_list.append(index)

        self._ctrl.current_camera_changed.emit(add)
    # end of accepted

# end of Js06CameraView

class Js06EditTarget(QDialog):
    def __init__(self, controller: Js06MainCtrl):
        super().__init__()

        ui_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "../resources/edit_target.ui")
        uic.loadUi(ui_path, self)
        self._ctrl = controller
        self._model = self._ctrl.get_target()

        self.target = []
        self.prime_x = []
        self.prime_y = []
        self.target_x = []
        self.target_y = []
        self.label_x = []
        self.label_y = []
        self.distance = []
        self.oxlist = []
        self.result = []
        self.target_process = False
        self.csv_path = None

        transform = QTransform().rotate(-180)
        image = self._ctrl.image.mirrored(True, False)
        self.image_label.setPixmap(QPixmap.fromImage(image.transformed(transform)))
        self.image_label.setMaximumSize(self.width(), self.height())

        self.w = image.width()
        self.h = image.height()

        self.blank_lbl = QLabel(self)

        self.blank_lbl.paintEvent = self.blank_paintEvent
        self.blank_lbl.mousePressEvent = self.blank_mousePressEvent
        self.buttonBox.accepted.connect(self.save_targets)
        self.buttonBox.rejected.connect(self.rejected_btn)

        self.numberCombo.currentIndexChanged.connect(self.combo_changed)

        self.blank_lbl.raise_()
        self.get_target()
        self.combo_changed()
    # end of __init__

    def combo_changed(self):
        targets = self._model
        for i in range(len(targets)):
            if self.numberCombo.currentText() == str(i + 1):
                self.labelEdit.setText(str(targets[i]['label']))
                self.ordinalEdit.setText(str(targets[i]['distance']))
                self.categoryEdit.setText(str(targets[i]['category']))
                self.distanceEdit.setText(str(targets[i]['distance']))
                self.coordinate_x_Edit.setText(str(targets[i]['roi']['point'][0]))
                self.coordinate_y_Edit.setText(str(targets[i]['roi']['point'][1]))
                # self.point_x_Edit.setText(targets['point'])
                # self.point_y_Edit.setText(str(targets['point'][i]))
    # end of combo_changed

    def save_targets(self):
        targets = self._model
        self.close()
    # end of save_targets

    def save_cameras(self):
        cameras = self._model.get_data()
        self._ctrl.update_cameras(cameras)
    # end of save_cameras

    def rejected_btn(self):
        self.close()
    # end of rejected_btn

    def blank_paintEvent(self, event):
        self.painter = QPainter(self.blank_lbl)
        if self.target:
            for name, x, y in zip(self.target, self.label_x, self.label_y):
                self.painter.drawRect(x - (25 / 4), y - (25 / 4), 25 / 2, 25 / 2)
                self.painter.drawText(x - 4, y - 10, f"{name}")
        self.blank_lbl.setGeometry(self.image_label.geometry())

        self.painter.end()
    # end of paintEvent

    def blank_mousePressEvent(self, event):
        x = int(event.pos().x() / self.width() * self.w)
        y = int(event.pos().y() / self.height() * self.h)

        for i in range(len(self.target)):
            self.target[i] = i + 1
        # for i in range(len(self.target)):
        #     if self.target_x[i] - 25 < x < self.target_x[i] + 25 and \
        #             self.target_y[i] - 25 < y < self.target_y[i] + 25:
        #         if self.oxlist[i] == 0:
        #             self.oxlist[i] = 1
        #         else:
        #             self.oxlist[i] = 0
        # if not self.target_process:
        #     return
        if event.buttons() == Qt.LeftButton:
            self.target = []
            text, ok = QInputDialog.getText(self, 'Add Target', 'Distance (km)')
            if ok and text:
                self.target_x.append(float(x))
                self.target_y.append(float(y))
                self.distance.append(float(text))
                self.target.append(str(len(self.target_x)))
                self.oxlist.append(0)
                print(f"Target position: {self.target_x[-1]}, {self.target_y[-1]}")
                # self.coordinator()
                self.save_target()
                self.get_target()

                self.numberCombo.clear()
                for i in range(len(self.target)):
                    print(i)
                    self.numberCombo.addItem(str(i + 1))
                    self.numberCombo.setCurrentIndex(i)
                    self.labelEdit.setText(f"t{i + 1}")
                    # self.coordinate_x_Edit.setText(str(round(self.prime_x[i], 2)))
                    # self.coordinate_y_Edit.setText(str(round(self.prime_y[i], 2)))
                self.distanceEdit.setText(text)
                self.ordinalEdit.setText("E")
                self.categoryEdit.setText("single")
                self.coordinate_x_Edit.setText(str(x))
                self.coordinate_y_Edit.setText(str(y))
            print(self.result)

        if event.buttons() == Qt.RightButton:
            # pylint: disable=invalid-name
            text, ok = QInputDialog.getText(self, 'Remove Target', 'Enter target number to remove')
            if ok and text:
                if len(self.target) >= 1:
                    text = int(text)
                    del self.target[text - 1]
                    del self.prime_x[text - 1]
                    del self.prime_y[text - 1]
                    del self.label_x[text - 1]
                    del self.label_y[text - 1]
                    del self.distance[text - 1]
                    del self.oxlist[text - 1]
                    print(f"[Target {text}] remove.")
    # end of label_mousePressEvent

    def coordinator(self):
        self.prime_y = [y / self.h for y in self.target_y]
        self.prime_x = [2 * x / self.w - 1 for x in self.target_x]
    # end of coordinator

    def restoration(self):
        self.target_x = [int((x + 1) * self.w / 2) for x in self.prime_x]
        self.target_y = [int(y * self.h) for y in self.prime_y]
    # end of restoration

    def save_target(self):
        if self.target:
            for i in range(len(self.target)):
                self.result[i]['label'] = self.target[i]
                self.result[i]['label_x'] = [int(x * self.width() / self.w) for x in self.target_x][i]
                self.result[i]['label_y'] = [int(y * self.height() / self.h) for y in self.target_y][i]
                self.result[i]["distance"] = self.distance

            # Save Target Information in mongoDB
    # end of save_target

    def get_target(self):
        targets = self._model

        self.numberCombo.clear()
        for i in range(len(targets)):
            self.numberCombo.addItem(str(i + 1))
        self.result = targets

        for i in range(len(targets)):
            self.target.append(self.result[i]['label'])
            self.label_x.append(self.result[i]['roi']['point'][0])
            self.label_y.append(self.result[i]['roi']['point'][1])
            self.distance.append(self.result[i]['distance'])
        print(f"target: {self.target}")
        print(f"label_x: {self.label_x}")
        print(f"label_y: {self.label_y}")
        print(f"distance: {self.distance}")

    # end of get_target

# end of Js06EditTarget

class Js06VideoWidget(QWidget):
    """Video stream player using QGraphicsVideoItem
    """
    grabImage = pyqtSignal(QImage)

    def __init__(self, parent: QObject):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.graphicView = QGraphicsView(self.scene)
        self.graphicView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._video_item = QGraphicsVideoItem()
        self.scene.addItem(self._video_item)

        self.player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self._video_item)
        self.player.setPosition(0)

        layout = QVBoxLayout(self)
        layout.addWidget(self.graphicView)
        self.uri = None
        self.cam_name = None

        self.probe = QVideoProbe(self)
        self.probe.videoFrameProbed.connect(self.on_videoFrameProbed)
        self.probe.setSource(self.player)
    # end of __init__

    @pyqtSlot(QVideoFrame)
    def on_videoFrameProbed(self, frame:QVideoFrame):
        self.grabImage.emit(frame.image())
    # end of on_videoFrameProbed

    def mousePressEvent(self, event: QMouseEvent):
        self.graphicView.fitInView(self._video_item)
    # end of mousePressEvent

    def draw_roi(self, point: tuple, size: tuple):
        """Draw a boundary rectangle of ROI
        Parameters:
          point: the upper left point of ROI in canonical coordinates
          size: the width and height of ROI in canonical coordinates
        """
        rectangle = QGraphicsRectItem(*point, *size, self._video_item)
        rectangle.setPen(QPen(Qt.blue))
    # end of draw_roi

    @pyqtSlot(QMediaPlayer.State)
    def on_stateChanged(self, state):
        if state == QMediaPlayer.PlayingState:
            self.view.fitInView(self._video_item, Qt.KeepAspectRatio)
    # end of on_stateChanged

    @pyqtSlot(str)
    def on_camera_change(self, uri):
        print("DEBUG:", uri)
        self.uri = uri
        self.player.setMedia(QMediaContent(QUrl(uri)))
        self.player.play()

        # self.graphicView.fitInView(self._video_item)

        # self.blank_lbl.paintEvent = self.paintEvent
        # self.blank_lbl.raise_()

        # if url == VIDEO_SRC3:
        #     self.camera_name = "XNO-8080R"
        # print(self.camera_name)
        # self.get_target()

        # self.video_thread = VideoThread(url)
        # self.video_thread.update_pixmap_signal.connect(self.convert_cv_qt)
        # self.video_thread.start()
    # end of on_camera_change

    # def restoration(self):
    #     try:
    #         if self.target:
    #             self.target_x = [round((x + 1) * self.video_item.nativeSize().width() / 2) for x in self.prime_x]
    #             self.target_y = [round(y * self.video_item.nativeSize().height()) for y in self.prime_y]
    #     except:
    #         print(traceback.format_exc())
    #         sys.exit()
    # # end of restoration

    @property
    def video_item(self):
        return self._video_item
    # end of video_item

# end of VideoWidget

class Js06MainView(QMainWindow):
    restore_defaults_requested = pyqtSignal()
    main_view_closed = pyqtSignal()
    select_camera_requested = pyqtSignal()

    def __init__(self, controller: Js06MainCtrl):
        super().__init__()

        ui_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "../resources/main_view.ui")
        uic.loadUi(ui_path, self)
        self._ctrl = controller

        # Connect signals and slots
        self.restore_defaults_requested.connect(self._ctrl.restore_defaults)
        self.actionSelect_Camera.triggered.connect(self.select_camera)
        
        # Check the exit status
        normal_exit = self._ctrl.check_exit_status()
        if not normal_exit:
            self.ask_restore_default()

        # self.showFullScreen()
        # self.setGeometry(400, 50, 1500, 1000)
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)

        self.actionEdit_Target.triggered.connect(self.edit_target)
        # self.actionSelect_Camera.triggered.connect(self.select_camera_triggered)

        self.video_dock = QDockWidget("Video", self)
        self.video_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable)
        self.video_widget = Js06VideoWidget(self)
        self.video_dock.setWidget(self.video_widget)
        self.setCentralWidget(self.video_dock)
        self.video_widget.grabImage.connect(self._ctrl.update_image)
        self._ctrl.current_camera_changed.connect(self.video_widget.on_camera_change)
        self._ctrl.current_camera_changed.emit(self._ctrl.get_current_camera_uri())

        # The parameters in the following codes is for the test purposes. 
        # They should be changed to use canonical coordinates.
        # self.video_widget.draw_roi((50, 50), (40, 40))
    # end of __init__

    # self.qtimer = QTimer()
    # self.qtimer.setInterval(2000)
    # self.qtimer.timeout.connect(self.video_widget.inference_clicked)
    # self.qtimer.start()

    # # target plot dock
    # self.target_plot_dock = QDockWidget("Target plot", self)
    # self.addDockWidget(Qt.BottomDockWidgetArea, self.target_plot_dock)
    # self.target_plot_dock.setFeatures(
    #     QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
    # self.target_plot_widget = Js06TargetPlotWidget2(self)
    # self.target_plot_dock.setWidget(self.target_plot_widget)

    # # grafana dock 1
    # self.web_dock_1 = QDockWidget("Grafana plot 1", self)
    # self.addDockWidget(Qt.BottomDockWidgetArea, self.target_plot_dock)
    # self.web_dock_1.setFeatures(
    #     QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
    # self.web_view_1 = Js06TimeSeriesPlotWidget()
    # self.web_dock_1.setWidget(self.web_view_1)

    # self.splitDockWidget(self.target_plot_dock, self.web_dock_1, Qt.Horizontal)
    # self.tabifyDockWidget(self.target_plot_dock, self.web_dock_1)
    # end of __init__

    # def select_camera(self):
    #     text, ok = QInputDialog.getItem(self, "Select Camera",
    #                                     "Select Camera Manufacturer", ["H", "C", "F"], 0, False)
    #     text1, ok1 = QInputDialog.getText(self, "Select Camera", "Input Camera URI")
    #     print(text1)
    #     if ok and ok1:
    #         if text == "H" and text1 is not None:
    #             SRC = f"rtsp://admin:sijung5520@{text1}/profile2/media.smp"
    #             self.video_widget.onCameraChange(SRC)
    # # end of select_cam

    def moveEvent(self, event: QMoveEvent):
        # print(self.geometry())
        pass
    # end of moveEvent

    def edit_target(self):
        self.video_widget.player.stop()

        uri = self._ctrl.get_current_camera_uri()

        dlg = Js06EditTarget(self._ctrl)
        dlg.exec_()
        self.video_widget.player.play()
    # end of edit_target

    @pyqtSlot()
    def select_camera(self):
        dlg = Js06CameraView(self._ctrl)
        dlg.exec_()
    # end of select_cameara

    def ask_restore_default(self):
        # Check the last shutdown status
        response = QMessageBox.question(
            self,
            'JS-06 Restore to defaults',
            'The JS-06 exited abnormally.'
            'Do you want to restore the factory default?',
        )
        if response == QMessageBox.Yes:
            self.restore_defaults_requested.emit()
    # end of ask_restore_default

    # TODO(kwchun): its better to emit signal and process at the controller
    def closeEvent(self, event: QCloseEvent):
        self._ctrl.set_normal_shutdown()
        # event.accept()
    # end of closeEvent

    # def inference(self):
    #     self.video_widget.graphicView.fitInView(self.video_widget.video_item)

    def target_mode(self):
        """Set target image modification mode"""
        # self.save_target()
        if self.target_process:
            print(self.target_process)
    # end of target_mode

    def open_with_rtsp(self):
        text, ok = QInputDialog.getText(self, "Input RTSP", "Only Hanwha Camera")
        if ok:
            print(text)
    # end of open_with_rtsp

# end of Js06MainView

# if __name__ == '__main__':
#     import sys
#     from PyQt5.QtWidgets import QApplication  # pylint: disable=no-name-in-module

#     app = QApplication(sys.argv)
#     window = Js06MainView()
#     window.show()
#     sys.exit(app.exec_())

# end of view.py
