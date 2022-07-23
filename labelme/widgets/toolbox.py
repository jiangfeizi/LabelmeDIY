from qtpy import QtCore
from qtpy import QtWidgets
from labelme.ui.tools_ui import Ui_Form


class Toolbox(QtWidgets.QWidget):
    def __init__(self):
        super(Toolbox, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    window = Toolbox()
    window.show()

    app.exec_()
