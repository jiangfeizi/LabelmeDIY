from qtpy import QtCore
from qtpy import QtWidgets
from labelme.ui.tools_ui import Ui_Form


class Tools(QtWidgets.QWidget):
    def __init__(self):
        super(Tools, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    window = Tools()
    window.show()

    app.exec_()
