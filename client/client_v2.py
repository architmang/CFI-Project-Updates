import sys
import os
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QMessageBox
from client_ui_v2 import Ui_MainWindow
import socket
import cv2
import numpy as np
import time
#import PySpin
import shutil


IP1 = "142.244.63.34" #"192.168.0.22"  #localhost"
IP2 = "142.244.63.34"
IP3 = "169.254.46.209"
IP4 = "169.254.64.101"
SERVER_NAME1 = "azure_kinect1"
SERVER_NAME2 = "azure_kinect2"
SERVER_NAME3 = "kinect_v2_1"
SERVER_NAME4 = "kinect_v2_2"
PORT1 = 27016
PORT2 = 27016
PORT3 = 27016
PORT4 = 27016

IPS = [IP1] #[IP1] 
SERVER_NAMES = [SERVER_NAME1]#, SERVER_NAME2] # [SERVER_NAME1]
PORTS  = [PORT1]#, PORT2] #[PORT1] 
MAX_FRAMES = 2000



class Connection:
    def __init__(self, name, ip, b_active, port):
        self.s = socket.socket()
        self.server_name = name
        self.ip = ip
        self.b_active = b_active  # whether check
        if b_active:
            print('try to connect to %s %s:%i' % (name, ip, port))
            self.s.connect((ip, port))
            print('connected to %s %s:%i' % (name, ip, port))
            buf = "device_name " + name
            self.s.send(bytes(buf, encoding='ascii'))
            info = self.s.recv(512).decode('ascii')
            print(info)

    def send(self, buf):
        if self.b_active:
            self.s.send(bytes(buf, encoding='ascii'))

    def recv(self, maxlen=512):
        if self.b_active:
            return self.s.recv(maxlen).decode('ascii')
        else:
            return None

    def send_calib_snap(self):
        if self.b_active:
            self.send('calib_snap %s_calib_snap' % self.server_name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.s.close()


class RecordThread(QThread):
    sinOut = pyqtSignal(str, str, str, str, str, str, str, str)

    def __init__(self, cons): 
        super(RecordThread, self).__init__()

        self.cons = cons

        self.record_num_frame = 0

        self.stop = False
        self.is_finish = False
        self.sub_path = None

    def run(self):
        assert self.sub_path is not None
        start_time = time.time()
        while self.record_num_frame < MAX_FRAMES:
            print(self.record_num_frame)
            # time.sleep(1. / 1000)  # second ~10fps
            if self.stop:
                break

            self.record_num_frame += 1
            msg = 'capture_one %s' % self.sub_path
            print(msg)
            for con in self.cons:
                con.send(msg)
            
            for con in self.cons:
                con.recv()

            # check other connections finishing capturing one frame
            idx_record_frames = ['0', '0', '0', '0']
            idx_calib_frames = ['0', '0', '0', '0']
            record_ones = ['0', '0', '0', '0']
            while self.record_num_frame < MAX_FRAMES:

                for i, con in zip(range(len(self.cons)), self.cons):
                    if con.b_active:
                        con.send('check')
                        print('record thread check')
                        _, _, idx_record_frames[i], idx_calib_frames[i], record_ones[i] = con.recv().split(' ')
                    else:
                        idx_record_frames[i], idx_calib_frames[i], record_ones[i] = '0', '0', '0'

                if (record_ones[0] == '0' and record_ones[1] == '0' and record_ones[2] == '0' and record_ones[3] == '0') or int(idx_record_frames[i])>=MAX_FRAMES:
                    break

            self.sinOut.emit(idx_record_frames[0], idx_calib_frames[0],
                             idx_record_frames[1], idx_calib_frames[1],
                             idx_record_frames[2], idx_calib_frames[2],
                             idx_record_frames[3], idx_calib_frames[3])
            if int(idx_record_frames[i])>=MAX_FRAMES:
                break

        end_time = time.time()
        print('number of frames recorded: %i' % self.record_num_frame)
        print('fps %.2f' % (self.record_num_frame / float(end_time - start_time)))

        self.is_finish = True

    def stop_running(self):
        self.stop = True
        print('record thread stops running')

    def enable_running(self):
        self.stop = False
        self.is_finish = False
        self.polar_timestamps = []
        self.polar_raw_imgs = []
        self.polar_idx_record_frame = 0
        self.record_num_frame = 0
        print('record thread initializes')

    def is_finish_recording(self):
        return self.is_finish

    def load_sub_path(self, sub_path):
        self.sub_path = sub_path

    def is_finish_load_from_buffer(self):
        return self.is_finish

    def empty_buffer(self):
        pass

    def clear_cam(self):
        pass

class SaveThread(QThread):
    sinOut = pyqtSignal(str, str, str, str, str, str, str, str)

    def __init__(self, cons): #save_dir, con1, con2, con3, con4, polar_active=False):
        super(SaveThread, self).__init__()

        self.cons = cons
        self.polar_idx_record_frame = 0
        self.sub_path = None
        self.if_loaded = False

    def run(self):
        if self.if_loaded:

            idx_record_frames = ['0', '0', '0', '0']
            idx_calib_frames = ['0', '0', '0', '0']
            record_ones = ['0', '0', '0', '0']

            while True:
                time.sleep(50. / 1000)  # second

                for i, con in zip(range(len(self.cons)), self.cons):
                    if con.b_active:
                        con.send('check')
                        print('Save thread: check')
                        _, _, idx_record_frames[i], idx_calib_frames[i], record_ones[i] = con.recv().split(' ')
                    else:
                        idx_record_frames[i], idx_calib_frames[i], record_ones[i] = '0', '0', '0'

                self.sinOut.emit(idx_record_frames[0], idx_calib_frames[0],
                                 idx_record_frames[1], idx_calib_frames[1],
                                 idx_record_frames[2], idx_calib_frames[2],
                                 idx_record_frames[3], idx_calib_frames[3])
                if idx_record_frames[0] == '0' and idx_record_frames[1] == '0' and \
                        idx_record_frames[2] == '0' and idx_record_frames[3] == '0':
                    break

            self.if_loaded = False
            self.polar_timestamps = []
            self.polar_raw_imgs = []
            print('finish saving')

    def load_data(self, polar_raw_imgs, polar_timestamps, sub_path):
        assert len(polar_raw_imgs) == len(polar_timestamps)
        self.polar_timestamps = polar_timestamps
        self.polar_raw_imgs = polar_raw_imgs
        self.polar_idx_record_frame = len(polar_raw_imgs)
        self.sub_path = sub_path
        self.if_loaded = True

class ClientDlg(QMainWindow, Ui_MainWindow):
    def __init__(self, names, idx_name=0, s_recording=0):
        QMainWindow.__init__(self)
        self.names = names
        self.idx_name = idx_name
        self.s_recording = s_recording  # flag to control stop recording or not
        self.setupUi(self)
        self.btnCalibSnap.setEnabled(False)
        self.btnRecord.setEnabled(False)
        self.btnRecordSave.setEnabled(False)
        self.txtName.setText(self.names[self.idx_name])

        self.cons = []
        self.con1, self.con2, self.con3, self.con4 = None, None, None, None
        self.record_thread = None
        self.save_thread = None
        self.infos = [self.info1, self.info2, self.info3, self.info4, self.info5]

    def show_image(self):
        # fname = 'example_image/' + str(self.txtName.text()) + '.jpg'
        # if not os.path.isfile(fname):
        #     fname = 'example_image/action.png'
        # myPixmap = QtGui.QPixmap(fname)
        # myScaledPixmap = myPixmap.scaled(self.lb_image.size(), QtCore.Qt.KeepAspectRatio)
        # self.lb_image.setPixmap(myScaledPixmap)
        pass

    def closeEvent(self, event=None):
        pass

    def on_exit(self):
        self.close()

    def on_remove(self):
        sub_path = self.txtName.text()
        qm = QMessageBox()
        ret = qm.question(self, '', "Are you sure to reset (remove) %s?" % sub_path, qm.Yes | qm.No)
        if ret == qm.Yes:
            # sub_path: all the content in the file "sub_path"
            for con in self.cons:
                print('remove rmdir')
                con.send('rmdir ' + str(sub_path))
                con.recv()

    def on_restart(self):
        self.idx_name = 0
        self.txtName.setText(self.names[self.idx_name])
        self.show_image()

    def on_next(self): 
        self.idx_name = min(self.idx_name+1, len(self.names)-1)
        self.txtName.setText(self.names[self.idx_name])
        self.show_image()

    def on_previous(self): 
        self.idx_name = max(self.idx_name-1, 0)
        self.txtName.setText(self.names[self.idx_name])
        self.show_image()

    def on_connect(self):
        self.btnConnect.setEnabled(False)
        self.ck1.setEnabled(False)
        self.ck2.setEnabled(False)
        self.ck3.setEnabled(False)
        self.ck4.setEnabled(False)
        checks = [self.ck1.isChecked(), self.ck2.isChecked(), self.ck3.isChecked(), self.ck4.isChecked()]
        # connect to server
        for i in range(len(IPS)):
            self.cons.append(Connection(SERVER_NAMES[i], IPS[i], checks[i], PORTS[i]))

        self.on_calib_mode()

    def on_calib_mode(self):
        self.btnCalibSnap.setEnabled(True)
        self.btnRecord.setEnabled(False)
        self.btnRecordSave.setEnabled(False)
        self.btnNext.setEnabled(False)
        self.btnPrevious.setEnabled(False)

    def on_calib_snap(self):
        # send calib snap message
        for con in self.cons:
            con.send_calib_snap()

        for con in self.cons:
            con.recv()

        idx_record_frames = ['', '', '', '']
        idx_calib_frames = ['', '', '', '']
        for i, con, info in zip(range(len(self.cons)), self.cons, self.infos):
            if con.b_active:
                con.send('check')
                print('calib snap check')
                _, _, idx_record_frames[i], idx_calib_frames[i], _ = con.recv().split(' ')
                info.setText('%s\n id record: %s\n id calib: %s' %
                               (SERVER_NAMES[i], idx_record_frames[i], idx_calib_frames[i]))
            else:
                info.setText('%s\n not connect' % SERVER_NAMES[i])
        
        # print("On calib_snap:", idx_record_frames, idx_calib_frames)

    def dis_info(self,
                 idx_record_frame1, idx_calib_frame1,
                 idx_record_frame2, idx_calib_frame2,
                 idx_record_frame3, idx_calib_frame3,
                 idx_record_frame4, idx_calib_frame4):

        idx_record_frames = [idx_record_frame1, idx_record_frame2, idx_record_frame3, idx_record_frame4]
        idx_calib_frames = [idx_calib_frame1, idx_calib_frame2, idx_calib_frame3, idx_calib_frame4]
  
        for i, con, info in zip(range(len(self.cons)), self.cons, self.infos):
            # print("On is_info:", i, len(self.cons), idx_record_frames[i], idx_record_frames)
            if con.b_active:
                info.setText('%s\n id record: %s\n id calib: %s' %
                               (SERVER_NAMES[i], idx_record_frames[i], idx_calib_frames[i]))
            else:
                info.setText('%s\n not connect' % SERVER_NAMES[i])

    def on_record_mode(self):
        self.btnCalibSnap.setEnabled(False)
        self.btnRecordSave.setEnabled(False)
        self.btnRecord.setEnabled(True)
        self.btnNext.setEnabled(True)
        self.btnPrevious.setEnabled(True)
        self.show_image()

        #print("On_record_mode")
        self.record_thread = RecordThread(self.cons) 
        self.record_thread.sinOut.connect(self.dis_info)
        self.save_thread = SaveThread(self.cons)
        self.save_thread.sinOut.connect(self.dis_info)
    
    def on_record(self):
        # "start Record" and "Stop Record"
        if not self.s_recording:
            # self.on_remove()
            msg = 'record_start ' + str(self.txtName.text())
            print(msg)
            for con in self.cons:
                con.send(msg)

            for con in self.cons:
                con.recv()

            self.record_thread.load_sub_path(str(self.txtName.text()))
            self.record_thread.enable_running()
            self.record_thread.start()

            self.s_recording = 1
            self.btnRecord.setText('Stop Record')
        else:
            # stop record and save buffered images and timestamps
            self.s_recording = 0
            sub_path = self.txtName.text()
            self.record_thread.stop_running()
            # wait until finish loading from the camera buffer
            while not self.record_thread.is_finish_load_from_buffer():
                time.sleep(100 / 1000)  # second
            self.save_thread.load_data(self.record_thread.polar_raw_imgs, self.record_thread.polar_timestamps, sub_path)
            self.record_thread.empty_buffer()
            self.save_thread.start()

            self.btnRecord.setText('Start Record')
            msg = 'record_stop'
            print(msg)
            for con in self.cons:
                con.send(msg)

            for con in self.cons:
                con.recv()

            self.on_next()

    def on_record_save_mode(self):
        self.btnCalibSnap.setEnabled(False)
        self.btnRecord.setEnabled(False)
        self.btnRecordSave.setEnabled(True)
        self.btnNext.setEnabled(True)
        self.btnPrevious.setEnabled(True)
        self.show_image()

        #print("On_record_save_mode")
        self.record_save_thread = RecordThread(self.cons)
        self.record_save_thread.sinOut.connect(self.dis_info)

    def on_record_save(self):
        # "start Record" and "Stop Record"
        if not self.s_recording:
            # self.on_remove()
            msg = 'record_save_start ' + str(self.txtName.text())
            print(msg)
            for con in self.cons:
                con.send(msg)

            for con in self.cons:
                con.recv()

            self.record_save_thread.load_sub_path(str(self.txtName.text()))
            self.record_save_thread.enable_running()
            self.record_save_thread.start()

            self.s_recording = 1
            self.btnRecordSave.setText('Stop Record and Save')
        else:
            self.s_recording = 0
            self.record_save_thread.stop_running()
            while not self.record_save_thread.is_finish_load_from_buffer():
                time.sleep(100 / 1000)  # second
            self.record_save_thread.empty_buffer()
            self.btnRecordSave.setText('Start Record and Save')
            msg = 'record_save_stop'
            print(msg) 
            for con in self.cons:
                con.send(msg)

            for con in self.cons:
                con.recv()

            self.on_next()

    def on_lineedit_enter(self):
        print(self.txtName.text())


def main(action_names):
    app = QApplication(sys.argv)
    window = ClientDlg(action_names)
    window.show()
    app.exec_()


if __name__ == '__main__':
    import os
    main(action_names=("1", "2", "3"))

