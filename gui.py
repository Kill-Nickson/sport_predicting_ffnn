import sys
import datetime

from PyQt5.QtCore import QThread, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget, QTextEdit, QTableWidget, QTableWidgetItem, \
    QTableView, QAbstractItemView, QHeaderView, QLabel, QPushButton

import opt_live_predictor as predictor


class Threader(QThread):
    authResult = pyqtSignal(object)
    updateStart = pyqtSignal(object)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.predictor_result = []

    def run(self):
        while True:
            self.updateStart.emit('Updating the results table...')

            predictions, matches_found, error_code = predictor.get_predicts_results()

            if matches_found is False:
                if error_code == 2:
                    self.authResult.emit(['Network currently down!', 2])
                elif error_code == 3:
                    self.authResult.emit(['Weights.json not found!', 3])
                elif error_code == 4:
                    self.authResult.emit(['Can\'t get active duty maps!', 4])
                else:
                    self.authResult.emit(['Matches are not found!', 0])
            else:
                self.authResult.emit(predictions)

            # A pause, preventing Error1015
            self.sleep(300)


class Window(QWidget):

    def __init__(self):
        super().__init__(flags=Qt.FramelessWindowHint)

        self.offset = None
        self.predictions = []
        self.push_button_quit = QPushButton(self)
        self.push_button_minimize = QPushButton(self)
        self.text_edit_title = QTextEdit(self)
        self.text_edit_matches_found = QTextEdit(self)
        self.text_edit_last_update_time = QTextEdit(self)
        self.table_widget = QTableWidget(0, 4, self)

        self.label = QLabel(self)
        self.label.setPixmap(QPixmap('./sport_predicting_ffnn/images/title_bar_icon.png'))
        self.label.resize(720, 50)
        self.label.move(0, -5)
        self.label.stackUnder(self.table_widget)

        self.label = QLabel(self)
        self.label.setPixmap(QPixmap('./sport_predicting_ffnn/images/title_icon.png'))
        self.label.resize(35, 35)
        self.label.move(5, 5)
        self.label.stackUnder(self.table_widget)

        self.bg_label = QLabel(self)
        self.bg_label.setPixmap(QPixmap('./sport_predicting_ffnn/images/bg_image.png'))
        self.bg_label.resize(720, 690)
        self.bg_label.move(0, 0)
        self.bg_label.stackUnder(self.label)
        self.bg_label.stackUnder(self.text_edit_last_update_time)
        self.bg_label.stackUnder(self.text_edit_matches_found)

        self.thread = Threader(self)
        self.thread.authResult.connect(self.handleAuthResult)
        self.thread.updateStart.connect(self.handleStartUpdate)
        self.thread.start()

        self.initUi()

        self.setWindowIcon(QIcon('./sport_predicting_ffnn/images/taskbar_icon.png'))
        self.setWindowTitle('The Predictor')
        self.setMinimumSize(QSize(720, 690))
        self.setMaximumSize(QSize(720, 690))

    def initUi(self):
        font = QFont('Calibri', 14)
        stylesheet = '''
                        QTextEdit#matches_found, QTextEdit#update_time {
                            color: rgb(154,157,161);
                            background-color: rgb(42,62,97);
                            border: none;
                        }
                        QTextEdit#title {
                            color: white;
                        }
                        QPushButton#quit {
                            border: none;
                        }
                        QPushButton#quit:hover {
                            background-image: url("./sport_predicting_ffnn/images/red-x.png");
                            background-repeat: no-repeat;
                        }
                        QPushButton#minimize {
                            border: none;
                        }
                        QPushButton#minimize:hover {
                            background-color: rgb(40, 67, 115);
                        }
                        QTableView::item
                        {
                            background: black;
                        }
                        '''

        layout = QGridLayout()
        layout.setSpacing(50)

        self.push_button_quit.setObjectName('quit')
        self.push_button_quit.move(678, 3)
        self.push_button_quit.resize(40, 40)
        self.push_button_quit.setStyleSheet(stylesheet)
        self.push_button_quit.stackUnder(self.table_widget)
        self.push_button_quit.setIcon(QIcon('./sport_predicting_ffnn/images/x.png'))
        self.push_button_quit.clicked.connect(self.button_quit)
        self.push_button_quit.setToolTip('Close app')
        self.push_button_quit.toolTip()

        self.push_button_minimize.setObjectName('minimize')
        self.push_button_minimize.move(640, 5)
        self.push_button_minimize.resize(34, 34)
        self.push_button_minimize.setStyleSheet(stylesheet)
        self.push_button_minimize.stackUnder(self.table_widget)
        self.push_button_minimize.setIcon(QIcon('./sport_predicting_ffnn/images/-.png'))
        self.push_button_minimize.clicked.connect(self.button_minimize)
        self.push_button_minimize.setToolTip('Minimize app\'s window')

        self.text_edit_title.setObjectName('title')
        self.text_edit_title.setText('The Predictor')
        self.text_edit_title.setFont(QFont('Calibri', 20))
        self.text_edit_title.setDisabled(True)
        self.text_edit_title.setStyleSheet('color:rgb(183, 206, 247);background: transparent;border: none;')
        self.text_edit_title.move(40, 0)
        self.text_edit_title.resize(270, 60)
        self.text_edit_title.stackUnder(self.table_widget)
        self.text_edit_title.show()

        self.text_edit_matches_found.setObjectName('matches_found')
        self.text_edit_matches_found.setFont(font)
        self.text_edit_matches_found.setDisabled(True)
        self.text_edit_matches_found.setStyleSheet(stylesheet)
        self.text_edit_matches_found.move(70, 68)
        self.text_edit_matches_found.resize(270, 40)
        self.text_edit_matches_found.setToolTip('Message box, which signalising about updating \n'
                                                'of the results table or showing reason \n'
                                                'why a results table is empty.')
        self.text_edit_matches_found.setToolTipDuration(5000)
        self.text_edit_matches_found.hide()

        self.text_edit_last_update_time.setObjectName('update_time')
        self.text_edit_last_update_time.setFont(font)
        self.text_edit_last_update_time.setDisabled(True)
        self.text_edit_last_update_time.setStyleSheet(stylesheet)
        self.text_edit_last_update_time.move(380, 68)
        self.text_edit_last_update_time.resize(270, 40)
        self.text_edit_last_update_time.setToolTip('Message box, showing a timestamp of last check or \n'
                                                   'update of a results table.')
        self.text_edit_last_update_time.setToolTipDuration(5000)
        self.text_edit_last_update_time.hide()

        self.setLayout(layout)
        layout.setContentsMargins(60, 130, 60, 60)

        self.table_widget.setHorizontalHeaderLabels(('Team 1', 'Team 2', 'Map', 'Prediction'))
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table_widget.horizontalHeader().setHighlightSections(False)
        self.table_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table_widget.horizontalHeader().setStyleSheet('::section{color:rgb(154,157,161); '
                                                           'background: rgb(24, 39, 66);}')
        self.table_widget.setStyleSheet('QTableView{background: rgb(31, 53, 89);border: none;}')

        for i in range(4):
            self.table_widget.setColumnWidth(i, 150)

        self.table_widget.setSelectionBehavior(QTableView.SelectRows)
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.table_widget.verticalHeader().setVisible(False)

        layout.addWidget(self.table_widget)

    def button_minimize(self):
        self.showMinimized()

    def button_quit(self):
        self.close()

    def handleAuthResult(self, result):
        """Function receives signals from a predicting thread and renew a result`s table"""

        self.table_widget.setRowCount(0)

        self.text_edit_last_update_time.setText(datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y'))
        self.text_edit_last_update_time.setAlignment(Qt.AlignCenter)
        self.text_edit_last_update_time.show()

        if len(result) == 2 and 'int' in str(type(result[1])):
            self.text_edit_matches_found.setText(result[0])
            self.text_edit_matches_found.setAlignment(Qt.AlignCenter)
            self.text_edit_matches_found.show()
        else:
            self.text_edit_matches_found.hide()

            for n, item in enumerate(result):
                split_item = item.split(':')
                self.table_widget.insertRow(n)

                r, g, b = (40, 90, 150) if n % 2 == 0 else (65, 85, 145)

                # Put predictions into the table
                for cell in range(4):
                    self.table_widget.setItem(n, cell, QTableWidgetItem(str(split_item[cell])))
                    self.table_widget.item(n, cell).setBackground(QColor(r, g, b))
                    self.table_widget.item(n, cell).setTextAlignment(Qt.AlignCenter)
                self.table_widget.item(n, 3).setToolTip('W1 means The Predictor predicts that "Team 1" will win, \n'
                                                        'otherwise W2 means The Predictor predicts \n'
                                                        'that "Team 2" will win.')

    def handleStartUpdate(self, result):
        """Signals a predictions update"""
        self.text_edit_matches_found.setText(result)
        self.text_edit_matches_found.show()

    # Next 3 mouse event handlers are used to allow a window dragging
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.offset = event.pos()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.offset is not None and event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen = Window()
    screen.show()
    sys.exit(app.exec_())
