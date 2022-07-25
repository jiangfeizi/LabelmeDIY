from qtpy import QtCore
from qtpy import QtWidgets


class Toolbox(QtWidgets.QWidget):
    def __init__(self):
        super(Toolbox, self).__init__()

        self.setWindowTitle("toolbox")
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setObjectName("gridLayout")

        self.labeled_button = QtWidgets.QPushButton(self)
        self.labeled_button.setObjectName("labeled")
        self.gridLayout.addWidget(self.labeled_button, 0, 0, 1, 1)
        self.unlabeled_button = QtWidgets.QPushButton(self)
        self.unlabeled_button.setObjectName("unlabeled")
        self.gridLayout.addWidget(self.unlabeled_button, 0, 1, 1, 1)
        self.rename_button = QtWidgets.QPushButton(self)
        self.rename_button.setObjectName("rename")
        self.gridLayout.addWidget(self.rename_button, 0, 2, 1, 1)
        self.classify_button = QtWidgets.QPushButton(self)
        self.classify_button.setObjectName("classify")
        self.gridLayout.addWidget(self.classify_button, 1, 0, 1, 1)
        self.filter_button = QtWidgets.QPushButton(self)
        self.filter_button.setObjectName("filter")
        self.gridLayout.addWidget(self.filter_button, 1, 1, 1, 1)
        self.cutout_button = QtWidgets.QPushButton(self)
        self.cutout_button.setObjectName("cutout")
        self.gridLayout.addWidget(self.cutout_button, 1, 2, 1, 1)

        self.labeled_button.setText("Labeled")
        self.unlabeled_button.setText("Unlabeled")
        self.rename_button.setText("Rename")
        self.classify_button.setText("Classify")
        self.filter_button.setText("Filter")
        self.cutout_button.setText("Cutout")

        self.labeled_button.clicked.connect(self.labeled)
        self.unlabeled_button.clicked.connect(self.unlabeled)
        self.rename_button.clicked.connect(self.rename)
        self.classify_button.clicked.connect(self.classify)
        self.filter_button.clicked.connect(self.filter)
        self.cutout_button.clicked.connect(self.cutout)

    def labeled(self):
        print("labeled")
        pass

    def unlabeled(self):
        print("unlabeled")
        pass

    def rename(self):
        print("rename")
        pass

    def classify(self):
        print("classify")
        pass

    def filter(self):
        print("filter")
        pass

    def cutout(self):
        print("cutout")
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    window = Toolbox()
    window.show()

    app.exec_()
