# !/usr/bin/env python3
#
# Copyright 2020-21 Sijung Co., Ltd.
# Authors: 
#     ruddyscent@gmail.com (Kyungwon Chun)
#     5jx2oh@gmail.com (Jongjin Oh)

from PyQt5.QtCore import QUrl, pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

VIDEO_SRC1 = "rtsp://admin:sijung5520@d617.asuscomm.com:1554/profile2/media.smp"
VIDEO_SRC2 = "rtsp://admin:sijung5520@d617.asuscomm.com:2554/profile2/media.smp"
VIDEO_SRC3 = "rtsp://admin:sijung5520@d617.asuscomm.com:3554/profile2/media.smp"

class Js06VideoWidget(QWidget):
    def __init__(self, parent=None):
        super(Js06VideoWidget, self).__init__(parent)
        self._viewer = QVideoWidget()  # self is required?
        self._player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self._player.setVideoOutput(self._viewer)
        # self._player.setMedia(QMediaContent(QUrl(VIDEO_SRC3)))
        self._player.setPosition(0)  # Required?
        self._viewer.show()
        # self._player.play()
        layout = QVBoxLayout(self)
        layout.addWidget(self._viewer)
        self._viewer.setGeometry(0, 0, 100, 100)

        self._viewer.mousePressEvent = self.viewer_mousePressEvent
    # end of __init__

    @pyqtSlot(str)
    def onCameraChange(self, url):
        self._player.setMedia(QMediaContent(QUrl(url)))
        self._player.play()
    # end of onCameraChange

    def viewer_mousePressEvent(self, event):
        print(event.pos())
        print(f"videoWidget: {self._viewer.sizeHint()}")
        # self.resize(self._viewer.sizeHint().width(), self._viewer.sizeHint().height())
        # self.resize(300, 300)
    # end of viewer_mousePressEvent


# end of VideoPlayer

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = Js06VideoWidget()
    window.show()
    sys.exit(app.exec_())
    
# end of video_widget.py
