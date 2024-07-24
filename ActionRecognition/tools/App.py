from argparse import ArgumentParser
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox, QTextEdit
)
from PyQt5.QtCore import Qt
from mmaction.apis.inferencers import MMAction2Inferencer
import sys

class MMAction2InferencerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('MMAction2 Inferencer')
        self.layout = QVBoxLayout()

        # Input video file or rawframes folder path
        self.layout.addWidget(QLabel('Input video file or rawframes folder path:'))
        self.inputPath = QLineEdit(self)
        self.inputPath.setPlaceholderText('C/bsh/Python/DL-Application/ActionRecognition/mmaction2/configs/recognition/abuse/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_abuse-rgb.py')
        self.layout.addWidget(self.inputPath)
        self.inputBrowse = QPushButton('Browse')
        self.inputBrowse.clicked.connect(self.browseInput)
        self.layout.addWidget(self.inputBrowse)

        # Output directory of videos
        self.layout.addWidget(QLabel('Output directory of videos:'))
        self.vidOutDir = QLineEdit(self)
        self.vidOutDir.setPlaceholderText('Enter output directory path')
        self.layout.addWidget(self.vidOutDir)
        self.outputBrowse = QPushButton('Browse')
        self.outputBrowse.clicked.connect(self.browseOutput)
        self.layout.addWidget(self.outputBrowse)

        # Pretrained action recognition algorithm config file
        self.recEnable = QCheckBox('Enable Config File')
        self.layout.addWidget(self.recEnable)
        self.recEnable.stateChanged.connect(self.toggleRec)
        self.rec = QLineEdit(self)
        self.rec.setPlaceholderText('Enter config file path')
        self.layout.addWidget(self.rec)
        self.configBrowse = QPushButton('Browse')
        self.configBrowse.clicked.connect(self.browseConfig)
        self.layout.addWidget(self.configBrowse)

        # Path to the custom checkpoint file of the selected recog model
        self.recWeightsEnable = QCheckBox('Enable Checkpoint File')
        self.layout.addWidget(self.recWeightsEnable)
        self.recWeightsEnable.stateChanged.connect(self.toggleWeights)
        self.recWeights = QLineEdit(self)
        self.recWeights.setPlaceholderText('Enter checkpoint file path')
        self.layout.addWidget(self.recWeights)
        self.weightsBrowse = QPushButton('Browse')
        self.weightsBrowse.clicked.connect(self.browseWeights)
        self.layout.addWidget(self.weightsBrowse)

        # Display the video in a popup window
        self.showVideo = QCheckBox('Display the video in a popup window')
        self.layout.addWidget(self.showVideo)

        # Whether to print the results
        self.printResult = QCheckBox('Whether to print the results')
        self.layout.addWidget(self.printResult)

        # File to save the inference results
        self.predOutEnable = QCheckBox('Enable Output File')
        self.layout.addWidget(self.predOutEnable)
        self.predOutEnable.stateChanged.connect(self.togglePredOut)
        self.predOutFile = QLineEdit(self)
        self.predOutFile.setPlaceholderText('Enter output file path')
        self.layout.addWidget(self.predOutFile)

        # Result display area
        self.resultDisplay = QTextEdit(self)
        self.resultDisplay.setReadOnly(True)
        self.layout.addWidget(self.resultDisplay)

        # Run button
        self.runButton = QPushButton('Run Inference')
        self.runButton.clicked.connect(self.runInference)
        self.layout.addWidget(self.runButton)

        self.setLayout(self.layout)

        self.toggleRec()
        self.toggleWeights()
        self.togglePredOut()

    def toggleRec(self):
        enabled = self.recEnable.isChecked()
        self.rec.setEnabled(enabled)
        self.configBrowse.setEnabled(enabled)

    def toggleWeights(self):
        enabled = self.recWeightsEnable.isChecked()
        self.recWeights.setEnabled(enabled)
        self.weightsBrowse.setEnabled(enabled)

    def togglePredOut(self):
        enabled = self.predOutEnable.isChecked()
        self.predOutFile.setEnabled(enabled)

    def browseInput(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Input File')
        self.inputPath.setText(path)

    def browseOutput(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        self.vidOutDir.setText(path)

    def browseConfig(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Config File')
        self.rec.setText(path)

    def browseWeights(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Checkpoint File')
        self.recWeights.setText(path)

    def runInference(self):
        args = {
            'inputs': self.inputPath.text(),
            'vid_out_dir': self.vidOutDir.text(),
            'show': self.showVideo.isChecked(),
            'print_result': self.printResult.isChecked(),
        }
        if self.recEnable.isChecked():
            args['rec'] = self.rec.text()
        if self.recWeightsEnable.isChecked():
            args['rec_weights'] = self.recWeights.text()
        if self.predOutEnable.isChecked():
            args['pred_out_file'] = self.predOutFile.text()

        init_args = {k: args[k] for k in ['rec', 'rec_weights'] if k in args}
        call_args = {k: args[k] for k in args if k not in init_args}
        mmaction2 = MMAction2Inferencer(**init_args)

        if self.printResult.isChecked():
            import io
            import contextlib

            # Capture the output of the inference
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                mmaction2(**call_args)
            result_text = output_buffer.getvalue()
            output_buffer.close()

            # Display the result in the QTextEdit widget
            self.resultDisplay.setText(result_text)
        else:
            mmaction2(**call_args)

class MMAction2InferencerApp:
    def __init__(self):
        self.parser = ArgumentParser()
        self.parser.add_argument('inputs', type=str, help='Input video file or rawframes folder path.')
        self.parser.add_argument('--vid-out-dir', type=str, default='', help='Output directory of videos.')
        self.parser.add_argument('--rec', type=str, default=None, help='Config file for the pretrained action recognition algorithm.')
        self.parser.add_argument('--rec-weights', type=str, default=None, help='Path to the custom checkpoint file.')
        self.parser.add_argument('--show', action='store_true', help='Display the video in a popup window.')
        self.parser.add_argument('--print-result', action='store_true', help='Whether to print the results.')
        self.parser.add_argument('--pred-out-file', type=str, default='', help='File to save the inference results.')

    def parse_args(self):
        call_args = vars(self.parser.parse_args())
        init_kws = ['rec', 'rec_weights']
        init_args = {init_kw: call_args.pop(init_kw) for init_kw in init_kws}
        return init_args, call_args

    def run(self):
        app = QApplication(sys.argv)
        ex = MMAction2InferencerUI()
        ex.show()
        sys.exit(app.exec_())

def main():
    app = MMAction2InferencerApp()
    app.run()

if __name__ == '__main__':
    main()
