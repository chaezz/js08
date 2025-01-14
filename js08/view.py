#!/usr/bin/env python3
#
# Copyright 2020-2021 Sijung Co., Ltd.
# 
# Authors: 
#     ruddyscent@gmail.com (Kyungwon Chun)
#     5jx2oh@gmail.com (Jongjin Oh)

import ast
import collections
import os
import sys
import vlc

from PyQt5 import uic
from PyQt5.QtChart import (QChart, QChartView, QDateTimeAxis, QLegend,
                           QLineSeries, QPolarChart, QScatterSeries,
                           QValueAxis)
from PyQt5.QtCore import (QDateTime, QObject, QPointF, Qt, pyqtSignal,
                          pyqtSlot)
from PyQt5.QtGui import QCloseEvent, QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QDialog, QFrame, QLabel, QMainWindow, QMessageBox,
                             QVBoxLayout, QWidget, QFileDialog)

from .controller import Js08MainCtrl
from .model import Js08Settings


class Js08DiscernmentView(QChartView):
    def __init__(self, parent: QWidget, title: str = None):
        super().__init__(parent)

        self.setRenderHint(QPainter.Antialiasing)

        chart = QPolarChart(title=title)
        chart.legend().setAlignment(Qt.AlignRight)
        chart.legend().setMarkerShape(QLegend.MarkerShapeCircle)
        self.setChart(chart)

        self.positives = QScatterSeries(name='Positive')
        self.negatives = QScatterSeries(name='Negative')
        self.positives.setColor(QColor('green'))
        self.negatives.setColor(QColor('red'))
        self.positives.setMarkerSize(10)
        self.negatives.setMarkerSize(10)
        chart.addSeries(self.positives)
        chart.addSeries(self.negatives)

        axis_x = QValueAxis()
        axis_x.setTickCount(9)
        axis_x.setRange(0, 360)
        axis_x.setLabelFormat('%d \xc2\xb0')
        axis_x.setTitleText('Azimuth (deg)')
        axis_x.setTitleVisible(False)
        chart.setAxisX(axis_x, self.positives)
        chart.setAxisX(axis_x, self.negatives)

        axis_y = QValueAxis()
        axis_y.setRange(0, 20)
        axis_y.setLabelFormat('%d')
        axis_y.setTitleText('Distance (km)')
        axis_y.setTitleVisible(False)
        chart.setAxisY(axis_y, self.positives)
        chart.setAxisY(axis_y, self.negatives)

    def keyPressEvent(self, event):
        keymap = {
            Qt.Key_Up: lambda: self.chart().scroll(0, -10),
            Qt.Key_Down: lambda: self.chart().scroll(0, 10),
            Qt.Key_Right: lambda: self.chart().scroll(-10, 0),
            Qt.Key_Left: lambda: self.chart().scroll(10, 0),
            Qt.Key_Greater: lambda: self.chart().zoonIn,
            Qt.Key_Less: lambda: self.chart().zoonOut,
        }
        callback = keymap.get(event.key())
        if callback:
            callback()

    @pyqtSlot(list, list)
    def refresh_stats(self, positives: list, negatives: list):
        pos_point = [QPointF(a, d) for a, d in positives]
        self.positives.replace(pos_point)
        neg_point = [QPointF(a, d) for a, d in negatives]
        self.negatives.replace(neg_point)


class Js08VisibilityView(QChartView):
    def __init__(self, parent: QWidget, maxlen: int, title: str = None):
        super().__init__(parent)

        now = QDateTime.currentSecsSinceEpoch()
        zeros = [(t * 1000, -1) for t in range(now - maxlen * 60, now, 60)]
        self.data = collections.deque(zeros, maxlen=maxlen)

        self.setRenderHint(QPainter.Antialiasing)

        chart = QChart(title=title)
        chart.legend().setVisible(False)
        self.setChart(chart)
        self.series = QLineSeries(name='Prevailing Visibility')
        chart.addSeries(self.series)

        axis_x = QDateTimeAxis()
        axis_x.setFormat('hh:mm')
        axis_x.setTitleText('Time')
        left = QDateTime.fromMSecsSinceEpoch(self.data[0][0])
        right = QDateTime.fromMSecsSinceEpoch(self.data[-1][0])
        axis_x.setRange(left, right)
        chart.setAxisX(axis_x, self.series)

        axis_y = QValueAxis()
        axis_y.setRange(0, 20)
        axis_y.setLabelFormat('%d')
        axis_y.setTitleText('Distance (km)')
        chart.setAxisY(axis_y, self.series)

        data_point = [QPointF(t, v) for t, v in self.data]
        self.series.append(data_point)

    def keyPressEvent(self, event):
        keymap = {
            Qt.Key_Up: lambda: self.chart().scroll(0, -10),
            Qt.Key_Down: lambda: self.chart().scroll(0, 10),
            Qt.Key_Right: lambda: self.chart().scroll(-10, 0),
            Qt.Key_Left: lambda: self.chart().scroll(10, 0),
            Qt.Key_Greater: lambda: self.chart().zoonIn,
            Qt.Key_Less: lambda: self.chart().zoonOut,
        }
        callback = keymap.get(event.key())
        if callback:
            callback()

    @pyqtSlot(int, dict)
    def refresh_stats(self, epoch: int, wedge_vis: dict):
        wedge_vis_list = list(wedge_vis.values())
        prev_vis = self.prevailing_visibility(wedge_vis_list)
        self.data.append((epoch * 1000, prev_vis))

        left = QDateTime.fromMSecsSinceEpoch(self.data[0][0])
        right = QDateTime.fromMSecsSinceEpoch(self.data[-1][0])
        self.chart().axisX().setRange(left, right)

        data_point = [QPointF(t, v) for t, v in self.data]
        self.series.replace(data_point)

    def prevailing_visibility(self, wedge_vis: list) -> float:
        if None in wedge_vis:
            return 0
        sorted_vis = sorted(wedge_vis, reverse=True)
        prevailing = sorted_vis[(len(sorted_vis) - 1) // 2]
        return prevailing


class Js08CameraView(QDialog):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setModal(True)

        if getattr(sys, 'frozen', False):
            directory = sys._MEIPASS
        else:
            directory = os.path.dirname(__file__)
        ui_path = os.path.join(directory, 'resources', 'camera_view.ui')
        uic.loadUi(ui_path, self)

        self._ctrl = parent._ctrl
        self._model = self._ctrl.get_camera_table_model()
        self.tableView.setModel(self._model)
        self.insertAbove.clicked.connect(self.insert_above)
        self.insertBelow.clicked.connect(self.insert_below)
        self.removeRows.clicked.connect(self.remove_rows)
        self.buttonBox.accepted.connect(self.accepted)

    def insert_above(self) -> None:
        selected = self.tableView.selectedIndexes()
        row = selected[0].row() if selected else 0
        self._model.insertRows(row, 1, None)

    def insert_below(self) -> None:
        selected = self.tableView.selectedIndexes()
        row = selected[-1].row() if selected else self._model.rowCount(None)
        self._model.insertRows(row + 1, 1, None)

    def remove_rows(self) -> None:
        selected = self.tableView.selectedIndexes()
        if selected:
            self._model.removeRows(selected[0].row(), len(selected), None)

    def accepted(self) -> None:
        # Update camera db
        cameras = self._model.get_data()
        self._ctrl.update_cameras(cameras, update_target=False)

        # Insert a new attr document, with new front_cam and rear_cam.
        front_cam = {}
        rear_cam = {}
        for cam in cameras:
            if cam['placement'] == 'front':
                uri = cam['uri']
                self._ctrl.front_camera_changed.emit(uri)
                front_cam = cam
                front_cam['camera_id'] = front_cam.pop('_id')
            elif cam['placement'] == 'rear':
                uri = cam['uri']
                self._ctrl.rear_camera_changed.emit(uri)
                rear_cam = cam
                rear_cam['camera_id'] = rear_cam.pop('_id')

        attr = self._ctrl.get_attr()
        attr['front_camera'] = front_cam
        attr['rear_camera'] = rear_cam
        del attr['_id']
        self._ctrl.insert_attr(attr)
        self._ctrl.front_camera_changed.emit(self._ctrl.get_front_camera_uri())
        self._ctrl.rear_camera_changed.emit(self._ctrl.get_rear_camera_uri())


class Js08TargetView(QDialog):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setModal(True)

        if getattr(sys, 'frozen', False):
            directory = sys._MEIPASS
        else:
            directory = os.path.dirname(__file__)
        ui_path = os.path.join(directory, 'resources', 'target_view.ui')
        uic.loadUi(ui_path, self)
        self._ctrl = parent._ctrl

        self.target = []
        self.target_x = []
        self.target_y = []
        self.point_x = []
        self.point_y = []
        self.distance = []
        self.size_x = []
        self.size_y = []
        self.result = []

        self.image = self._ctrl.grab_image('front')
        self.w = self.image.width()
        self.h = self.image.height()
        self.image_label.setPixmap(QPixmap.fromImage(self.image).scaled(self.w, self.h, Qt.KeepAspectRatio))
        self.image_label.setMaximumSize(self.width(), self.height())

        self.blank_lbl = QLabel(self)

        self.blank_lbl.mousePressEvent = self.blank_mousePressEvent
        self.buttonBox.accepted.connect(self.save_target)
        self.buttonBox.rejected.connect(self.rejected_btn)
        self.switch_btn.clicked.connect(self.switch_button)
        self.azimuth_check.stateChanged.connect(self.check)
        self.frame_direction = 0

        self.numberCombo.currentIndexChanged.connect(self.combo_changed)
        self.blank_lbl.raise_()

        self.get_target("front")

    def check(self) -> None:
        self.update()

    def switch_button(self):
        if self.frame_direction >= 2:
            self.frame_direction = 0

        if self.frame_direction == 0:
            self.image = self._ctrl.grab_image('rear')
            self.get_target("rear")
        else:
            self.image = self._ctrl.grab_image('front')
            self.get_target("front")

        self.w = self.image.width()
        self.h = self.image.height()
        self.image_label.setPixmap(QPixmap.fromImage(self.image).scaled(self.w, self.h, Qt.KeepAspectRatio))

        self.frame_direction = self.frame_direction + 1

        self.update()

    def combo_changed(self) -> None:
        self.blank_lbl.paintEvent = self.blank_paintEvent

        for i in range(len(self.target)):
            if self.numberCombo.currentText() == self.target[i]:
                # self.labelEdit.setText(str(self.target[i]))
                self.labelEdit.setText(self.numberCombo.currentText())
                self.distanceEdit.setText(str(self.distance[i]))
                self.point_x_Edit.setText(str(self.point_x[i]))
                self.point_y_Edit.setText(str(self.point_y[i]))
                break
            else:
                self.labelEdit.setText("")
                self.distanceEdit.setText("")
                self.point_x_Edit.setText("")
                self.point_y_Edit.setText("")
        self.update()

    def save_cameras(self) -> None:
        cameras = self._model.get_data()
        self._ctrl.update_cameras(cameras)

    def rejected_btn(self) -> None:
        self.close()

    def blank_paintEvent(self, event):
        self.painter = QPainter(self.blank_lbl)

        self.painter.setPen(QPen(Qt.white, 1, Qt.DotLine))
        if self.azimuth_check.isChecked():
            x1 = self.painter.drawLine(self.blank_lbl.width() * (1 / 4), 0,
                                       self.blank_lbl.width() * (1 / 4), self.blank_lbl.height())
            x2 = self.painter.drawLine(self.blank_lbl.width() * (1 / 2), 0,
                                       self.blank_lbl.width() * (1 / 2), self.blank_lbl.height())
            x3 = self.painter.drawLine(self.blank_lbl.width() * (3 / 4), 0,
                                       self.blank_lbl.width() * (3 / 4), self.blank_lbl.height())

        self.painter.setPen(QPen(Qt.red, 2))
        for name, x, y, sx, sy in zip(self.target, self.coor_point_x, self.coor_point_y, self.size_x, self.size_y):
            self.painter.drawRect(int(x - (25 / 4)), int(y - (25 / 4)), int(sx), int(sy))
            self.painter.drawText(x - 4, y - 10, f"{name}")

        self.blank_lbl.setGeometry(self.image_label.geometry())

        self.painter.end()

    def blank_mousePressEvent(self, event) -> None:
        x = int(event.pos().x() / self.blank_lbl.width() * self.w)
        y = int(event.pos().y() / self.blank_lbl.height() * self.h)

        for i in range(len(self.target)):
            self.target[i] = i + 1

        if event.buttons() == Qt.LeftButton:
            maxVal = max(self.target)

            self.numberCombo.addItem(str(maxVal + 1))
            self.numberCombo.setCurrentIndex(maxVal)

            self.point_x.append(int(event.pos().x()))
            self.point_y.append(int(event.pos().y()))
            self.size_x.append(30)
            self.size_y.append(30)

            self.target.append(len(self.point_x))
            self.distance.append(0)

            self.combo_changed()
            self.coordinator()

            print(f'Mouse press event - Add "{len(self.target)}th" target')
            # self.save_target()
            # self.get_target('front')

        if event.buttons() == Qt.RightButton:
            deleteIndex = self.numberCombo.currentIndex() + 1
            reply = QMessageBox.question(self, 'Delete Target',
                                         f'Are you sure delete target [{deleteIndex}] ?')
            if reply == QMessageBox.Yes:
                self.numberCombo.removeItem(deleteIndex - 1)
                self.numberCombo.setCurrentIndex(deleteIndex - 2)

                del self.point_x[deleteIndex - 1]
                del self.point_y[deleteIndex - 1]
                del self.target[deleteIndex - 1]
                self.combo_changed()

        self.update()

    def coordinator(self) -> None:
        self.prime_y = [y / self.h for y in self.target_y]
        self.prime_x = [2 * x / self.w - 1 for x in self.target_x]

    def restoration(self) -> None:
        self.target_x = [int((x + 1) * self.w / 2) for x in self.prime_x]
        self.target_y = [int(y * self.h) for y in self.prime_y]

    def save_target(self) -> None:
        label = self.labelEdit.text()
        distance = ast.literal_eval(self.distanceEdit.text())
        x = int(self.point_x_Edit.text())
        y = int(self.point_y_Edit.text())

        self.combo_changed()

        _id = self._ctrl.get_attr()
        if self.frame_direction == 0:
            _id = _id['front_camera']['camera_id']
        else:
            _id = _id['rear_camera']['camera_id']

        info = [{
            "_id": _id,
            "targets": [{
                "label": label,
                "distance": distance,
                # "mask": [f"{int(self.labelEdit.text()) + 1}-1.png", f"{int(self.labelEdit.text()) + 1}-2.png"],
                "roi": {"point": [x, y]}
            }]
        }]

        # TODO(Kyungwon): update camera db only, the current camera selection is
        # performed at camera view
        # Save Target through controller
        self._ctrl.update_cameras(info, True)
        print(info)

        self.close()

    def get_target(self, direction: str) -> None:
        # Initialize variable
        # This is because variables overlap
        self.target = []
        self.point_x = []
        self.point_y = []
        self.distance = []
        self.size_x = []
        self.size_y = []

        self.result = self._ctrl.get_target(direction)

        self.numberCombo.clear()
        for i in range(len(self.result)):
            self.target.append(self.result[i]['label'])
            self.point_x.append(self.result[i]['roi']['point'][0])
            self.point_y.append(self.result[i]['roi']['point'][1])
            self.distance.append(self.result[i]['distance'])
            self.size_x.append(self.result[i]['roi']['size'][0])
            self.size_y.append(self.result[i]['roi']['size'][1])

        self.numberCombo.addItems(self.target)

        # 1053 is self.image_label.width
        # 646 is self.image_label.height
        self.coor_point_x = [int(x * 1053 / self.w) for x in self.point_x]
        self.coor_point_y = [int(y * 646 / self.h) for y in self.point_y]
        self.coor_size_x = [int(x * 1053 / self.w) for x in self.size_x]
        self.coor_size_y = [int(y * 646 / self.h) for y in self.size_y]


class Js08AboutView(QDialog):
    def __init__(self) -> None:
        super().__init__()

        if getattr(sys, 'frozen', False):
            directory = sys._MEIPASS
        else:
            directory = os.path.dirname(__file__)
        ui_path = os.path.join(directory, 'resources', 'about_view.ui')
        uic.loadUi(ui_path, self)


class Js08ConfigView(QDialog):
    def __init__(self) -> None:
        super().__init__()

        if getattr(sys, 'frozen', False):
            directory = sys._MEIPASS
        else:
            directory = os.path.dirname(__file__)
        ui_path = os.path.join(directory, 'resources', 'config_view.ui')
        uic.loadUi(ui_path, self)

        self.image_base_path = None

        self.buttonBox.accepted.connect(self.write_values)

        self.read_values()

    def read_values(self) -> None:
        self.SaveVista_comboBox.setCurrentText(f"{Js08Settings.get('save_vista')}")
        self.SaveImagePatch_comboBox.setCurrentText(f"{Js08Settings.get('save_target_clip')}")
        self.ImageBasePath_pushButton.clicked.connect(self.ask_image_base_path)
        self.InferenceBatchSize_spinBox.setValue(Js08Settings.get('inference_batch_size'))
        self.DatabaseHost_lineEdit.setText(Js08Settings.get('db_host'))
        self.DatabasePort_lineEdit.setText(f"{Js08Settings.get('db_port')}")
        self.DatabaseName_lineEdit.setText(Js08Settings.get('db_name'))
        self.DatabaseAdmin_lineEdit.setText(Js08Settings.get('db_admin'))
        self.DatabaseAdminPw_lineEdit.setText(Js08Settings.get('db_admin_password'))
        self.DatabaseUser_lineEdit.setText(Js08Settings.get('db_user'))
        self.DatabaseUserPw_lineEdit.setText(Js08Settings.get('db_user_password'))

    def ask_image_base_path(self) -> None:
        self.image_base_path = QFileDialog.getExistingDirectory(self, "Select directory",
                                                                directory=Js08Settings.get('image_base_path'))

    def write_values(self) -> None:
        Js08Settings.set('save_vista', self.SaveVista_comboBox.currentText())
        Js08Settings.set('save_target_clip', self.SaveImagePatch_comboBox.currentText())
        if self.image_base_path is not None:
            Js08Settings.set('image_base_path', self.image_base_path)
        Js08Settings.set('inference_batch_size', self.InferenceBatchSize_spinBox.value())
        Js08Settings.set('db_host', self.DatabaseHost_lineEdit.text())
        Js08Settings.set('db_port', f"{int(self.DatabasePort_lineEdit.text())}")
        Js08Settings.set('db_name', self.DatabaseName_lineEdit.text())
        Js08Settings.set('db_admin', self.DatabaseAdmin_lineEdit.text())
        Js08Settings.set('db_admin_password', self.DatabaseAdminPw_lineEdit.text())
        Js08Settings.set('db_user', self.DatabaseUser_lineEdit.text())
        Js08Settings.set('db_user_password', self.DatabaseUserPw_lineEdit.text())


class Js08VideoWidget(QWidget):
    """Video stream player using QVideoWidget
    """
    def __init__(self, parent: QObject = None):
        super().__init__(parent)

        self.instance = vlc.Instance()
        self.mediaplayer = self.instance.media_player_new()
    
        self.videoframe = QFrame()
        
        if sys.platform.startswith("linux"):  # for Linux using the X Server
            self.mediaplayer.set_xwindow(self.videoframe.winId())
        elif sys.platform == "win32":  # for Windows
            self.mediaplayer.set_hwnd(self.videoframe.winId())
        elif sys.platform == "darwin":  # for MacOS
            self.mediaplayer.set_nsobject(self.videoframe.winId())

        layout = QVBoxLayout(self)
        layout.addWidget(self.videoframe)

    @pyqtSlot(str)
    def on_camera_change(self, uri: str):
        self.uri = uri
        media = self.instance.media_new(self.uri)
        self.mediaplayer.set_media(media)
        self.mediaplayer.play()


class Js08MainView(QMainWindow):
    restore_defaults_requested = pyqtSignal()
    main_view_closed = pyqtSignal()
    select_camera_requested = pyqtSignal()

    def __init__(self, controller: Js08MainCtrl, size: list = None):
        super().__init__()

        if getattr(sys, 'frozen', False):
            directory = sys._MEIPASS
        else:
            directory = os.path.dirname(__file__)
        ui_path = os.path.join(directory, 'resources', 'main_view.ui')
        uic.loadUi(ui_path, self)
        self._ctrl = controller

        # Connect signals and slots
        self.restore_defaults_requested.connect(self._ctrl.restore_defaults)

        # Check the exit status
        normal_exit = self._ctrl.check_exit_status()
        if not normal_exit:
            self.ask_restore_default()

        self.actionEdit_Camera.triggered.connect(self.edit_camera)
        self.actionEdit_Target.triggered.connect(self.edit_target)
        self.actionConfiguration.triggered.connect(self.configuration)
        self.actionAbout.triggered.connect(self.about_view)

        # Set size of Js08MainView
        if size == None:
            width, height = Js08Settings.get('window_size')
        else:
            width, height = size
        self.resize(width, height)

        # Front video
        self.front_video_widget = Js08VideoWidget(self)
        self.front_vertical.addWidget(self.front_video_widget, 1)
        self._ctrl.front_camera_changed.connect(self.front_video_widget.on_camera_change)
        self._ctrl.front_camera_changed.emit(self._ctrl.get_front_camera_uri())

        # Rear video
        self.rear_video_widget = Js08VideoWidget(self)
        self.rear_vertical.addWidget(self.rear_video_widget, 1)
        self._ctrl.rear_camera_changed.connect(self.rear_video_widget.on_camera_change)
        self._ctrl.rear_camera_changed.emit(self._ctrl.get_rear_camera_uri())

        # Discernment status
        self.discernment_widget = Js08DiscernmentView(self)
        self.discernment_vertical.addWidget(self.discernment_widget)
        self._ctrl.target_assorted.connect(self.discernment_widget.refresh_stats)

        # Prevailing visibility
        self.visibility_widget = Js08VisibilityView(self, 1440)
        self.visibility_vertical.addWidget(self.visibility_widget)
        self._ctrl.wedge_vis_ready.connect(self.visibility_widget.refresh_stats)

        self.show()

    def edit_target(self) -> None:
        dlg = Js08TargetView(self)
        dlg.resize(self.width(), self.height())
        dlg.exec_()

    def configuration(self) -> None:
        dlg = Js08ConfigView()
        dlg.exec_()

    def about_view(self) -> None:
        dlg = Js08AboutView()
        dlg.setFixedSize(dlg.size())
        dlg.exec_()

    @pyqtSlot()
    def edit_camera(self) -> None:
        dlg = Js08CameraView(self)
        dlg.exec()

    def ask_restore_default(self) -> None:
        # Check the last shutdown status
        response = QMessageBox.question(
            self,
            'Restore to defaults',
            'The JS-08 exited abnormally. '
            'Do you want to restore the factory default?',
        )
        if response == QMessageBox.Yes:
            self.restore_defaults_requested.emit()

    # TODO(kwchun): its better to emit signal and process at the controller
    def closeEvent(self, event: QCloseEvent) -> None:
        # Save currnet window size
        window_size = self.size()
        width = window_size.width()
        height = window_size.height()
        Js08Settings.set('window_size', (width, height))

        self._ctrl.set_normal_shutdown()


if __name__ == '__main__':
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = Js08MainView(Js08MainCtrl)
    sys.exit(app.exec())
