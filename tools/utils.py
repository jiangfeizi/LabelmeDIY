


def is_image(image_name):
    return image_name.endswith('.png') or image_name.endswith('.jpg') or image_name.endswith('.bmp') or image_name.endswith('.jpeg') or image_name.endswith('.tiff')


# self.tools_output_dir = os.path.join(os.getcwd(), 'output')

# clean = action(
#     self.tr("&Clean"),
#     self.clean_tools_output_dir,
#     icon="clean",
#     tip=self.tr("clean output directory of tools.")
# )

# valid = action(
#     self.tr("&Valid"),
#     self.output_valid,
#     icon="valid",
#     tip=self.tr("clean output directory of tools.")
# )

# separate = action(
#     self.tr("&Separate"),
#     self.separate_images,
#     icon="separate",
#     tip=self.tr("separate images to directories."),
# )


# def clean_tools_output_dir(self):
#     mb = QtWidgets.QMessageBox
#     msg = self.tr(
#         "You are about to clean output directory, "
#         "proceed anyway?"
#     )
#     answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
#     if answer != mb.Yes:
#         return

#     if not os.path.exists(self.tools_output_dir):
#         os.mkdir(self.tools_output_dir)
        
#     output_list = os.listdir(self.tools_output_dir)
#     count = len(output_list)

#     if count:
#         progress = QtWidgets.QProgressDialog(self)
#         progress.setWindowTitle(self.tr("Please wait a moment."))  
#         progress.setLabelText(self.tr("cleaning..."))
#         progress.setCancelButtonText(self.tr("cancel"))
#         progress.setMinimumDuration(3)
#         progress.setWindowModality(Qt.WindowModal)
#         progress.setRange(0,count)

#         for i in range(count):
#             progress.setValue(i) 
#             if progress.wasCanceled():
#                 QtWidgets.QMessageBox.warning(self, self.tr("Tips"), self.tr("Failed")) 
#                 break

#             item = output_list[i]
#             if os.path.isdir(os.path.join(self.tools_output_dir, item)):
#                 shutil.rmtree(os.path.join(self.tools_output_dir, item))
#             else:
#                 os.remove(os.path.join(self.tools_output_dir, item))
#         else:
#             progress.setValue(count) 
#             QtWidgets.QMessageBox.information(self, self.tr("Tips"), self.tr("Successed"))
#     else:
#         QtWidgets.QMessageBox.information(self, self.tr("Tips"), self.tr("Successed"))

# def tools_output_check(self):
#     if os.path.exists(self.tools_output_dir):
#         if len(os.listdir(self.tools_output_dir)):
#             QtWidgets.QMessageBox.warning(self, self.tr("Tips"), self.tr("Please clear output folder.")) 
#             return False
#     else:
#         QtWidgets.QMessageBox.warning(self, self.tr("Tips"), self.tr("Output folder has been created.")) 
#         os.mkdir(self.tools_output_dir)

#     return True

# def tools_opendir_check(self):
#     for item in os.listdir(self.lastOpenDir):
#         if os.path.isdir(os.path.join(self.lastOpenDir, item)):
#             QtWidgets.QMessageBox.warning(self, self.tr("Tips"), self.tr("The tools cann't support multilevel directory.")) 
#             return False
#     else:
#         return True

# def output_valid(self):
#     if not self.tools_opendir_check():
#         return
#     if self.tools_output_check():
#         count = self.fileListWidget.count()

#         if count:
#             progress = QtWidgets.QProgressDialog(self)
#             progress.setWindowTitle(self.tr("Please wait a moment."))  
#             progress.setLabelText(self.tr("Outputing files..."))
#             progress.setCancelButtonText(self.tr("cancel"))
#             progress.setMinimumDuration(3)
#             progress.setWindowModality(Qt.WindowModal)
#             progress.setRange(0,count) 

#             for i in range(count):
#                 progress.setValue(i) 
#                 if progress.wasCanceled():
#                     QtWidgets.QMessageBox.warning(self, self.tr("Tips"), self.tr("Failed")) 
#                     break

#                 item = self.fileListWidget.item(i)
#                 image_path = item.text()
#                 image_name = os.path.basename(image_path)
#                 pre, ext = os.path.splitext(image_path)
#                 label_path = pre + ".json"
#                 label_name = os.path.basename(label_path)
#                 if self.output_dir:
#                     label_path = os.path.join(self.output_dir, os.path.basename(label_path))

#                 if os.path.exists(label_path):
#                     data = json.load(open(label_path, 'r', encoding='utf8'))
#                     flags = data['flags']
#                     shapes = data['shapes']
#                     flag_list = [flag for key, flag in flags.items()]

#                     if sum(flag_list) >= 1 or len(shapes):
#                         shutil.copy(image_path, os.path.join(self.tools_output_dir, image_name))
#                         shutil.copy(label_path, os.path.join(self.tools_output_dir, label_name))
#             else:
#                 progress.setValue(count) 
#                 QtWidgets.QMessageBox.information(self, self.tr("Tips"), self.tr("Successed"))
#         else:
#             QtWidgets.QMessageBox.information(self, self.tr("Tips"), self.tr("Successed"))

# def separate_images(self):
#     if not self.tools_opendir_check():
#         return
#     if self.tools_output_check():
#         count = self.fileListWidget.count()

#         if count:
#             progress = QtWidgets.QProgressDialog(self)
#             progress.setWindowTitle(self.tr("Please wait a moment."))  
#             progress.setLabelText(self.tr("checking..."))
#             progress.setCancelButtonText(self.tr("cancel"))
#             progress.setMinimumDuration(3)
#             progress.setWindowModality(Qt.WindowModal)
#             progress.setRange(0,count) 

#             for i in range(count):
#                 progress.setValue(i) 
#                 if progress.wasCanceled():
#                     QtWidgets.QMessageBox.warning(self, self.tr("Tips"), self.tr("Failed")) 
#                     return

#                 item = self.fileListWidget.item(i)
#                 image_path = item.text()
#                 pre, ext = os.path.splitext(image_path)
#                 label_path = pre + ".json"
#                 if self.output_dir:
#                     label_path = os.path.join(self.output_dir, os.path.basename(label_path))

#                 if os.path.exists(label_path):
#                     data = json.load(open(label_path, 'r', encoding='utf8'))
#                     flags = data['flags']
#                     flag_list = [flag for key, flag in flags.items()]
#                     if sum(flag_list) >= 2:
#                         QtWidgets.QMessageBox.warning(self, self.tr("Tips"), self.tr("%s's flag is more than one.") % (image_path))
#                         return

#             for i in range(self.fileListWidget.count()):
#                 progress.setValue(i) 
#                 if progress.wasCanceled():
#                     QtWidgets.QMessageBox.warning(self, self.tr("Tips"), self.tr("Failed")) 
#                     return
                    
#                 item = self.fileListWidget.item(i)
#                 image_path = item.text()
#                 pre, ext = os.path.splitext(image_path)
#                 label_path = pre + ".json"
#                 if self.output_dir:
#                     label_path = os.path.join(self.output_dir, os.path.basename(label_path))

#                 if os.path.exists(label_path):
#                     data = json.load(open(label_path, 'r', encoding='utf8'))
#                     flags = data['flags']
#                     for key, flag in flags.items():
#                         if flag:
#                             key_dir = os.path.join(self.tools_output_dir ,key)
#                             if not os.path.exists(key_dir):
#                                 os.mkdir(key_dir)
#                             shutil.copy(image_path, os.path.join(key_dir, os.path.basename(image_path)))
#             else:
#                 progress.setValue(count) 
#                 QtWidgets.QMessageBox.information(self, self.tr("Tips"), self.tr("Successed"))
#         else:
#             QtWidgets.QMessageBox.information(self, self.tr("Tips"), self.tr("Successed"))                        

