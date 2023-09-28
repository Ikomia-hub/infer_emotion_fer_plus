from ikomia import utils, core, dataprocess
import copy
import os
import cv2
import numpy as np

SAMPLE_SIZE = 64


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class EmotionFerPlusParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.update = False
        self.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        self.target = cv2.dnn.DNN_TARGET_CPU
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)) , "models", "model.onnx")

    def set_values(self, param_map):
        # Set parameters values from Imageez application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(paramMap["windowSize"])
        pass

    def get_values(self):
        # Send parameters values to Imageez application
        # Create the specific dict structure (string container)
        param_map = {}
        # Example : paramMap["windowSize"] = str(self.windowSize)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class EmotionFerPlus(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add graphics output
        self.add_output(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.add_output(dataprocess.CNumericIO())

        # Network members
        self.net = None
        self.class_names = []

        # Create parameters class
        if param is None:
            self.set_param_object(EmotionFerPlusParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        # Load class names
        model_folder = os.path.dirname(os.path.realpath(__file__)) + "/models"
        with open(model_folder + "/class_names") as f:
            for row in f:
                self.class_names.append(row[:-1])

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Imageez application
        return 4

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Forward input image
        self.forward_input_image(0, 0)

        # Get parameters :
        param = self.get_param_object()

        # Load the recognition model from disk
        if self.net is None or param.update:
            if not os.path.exists(param.model_path):
                print("Downloading model, please wait...")
                model_url = utils.get_model_hub_url() + "/" + self.name + "/model.onnx"
                self.download(model_url, param.model_path)

            self.net = cv2.dnn.readNet(param.model_path)
            self.net.setPreferableBackend(param.backend)
            self.net.setPreferableTarget(param.target)
            param.update = False

        # Step progress bar:
        self.emit_step_progress()

        # Get input :
        input_img = self.get_input(0)
        input_graphics = self.get_input(1)

        # Init graphics output
        graphics_output = self.get_output(1)
        graphics_output.set_new_layer("EmotionFerPlus")
        graphics_output.set_image_index(0)

        # Init numeric output
        numeric_ouput = self.get_output(2)
        numeric_ouput.clear_data()
        numeric_ouput.set_output_type(dataprocess.NumericOutputType.TABLE)

        # Step progress bar:
        self.emit_step_progress()

        # Get image from input (numpy array):
        src_image = input_img.get_image()
        if src_image.ndim == 3:
            gray_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = src_image

        # Step progress bar:
        self.emit_step_progress()

        # Predict
        if input_graphics.is_data_available():
            index = 1
            items = input_graphics.get_items()

            for item in items:
                if item.get_type() == core.GraphicsItem.RECTANGLE:
                    x = int(item.x)
                    y = int(item.y)
                    w = int(item.width)
                    h = int(item.height)
                    face_img = gray_image[y:y+h, x:x+w]
                    emotion = self.predict(face_img, "Face #"+ str(index))
                    graphics_output.add_text(emotion, x + (0.05*w), y + (0.05*h))
                    index += 1   
        else:
            (h,w) = gray_image.shape
            emotion = self.predict(gray_image, "Full image")
            graphics_output.add_text(emotion, 0.05*w, 0.05*h)     

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process       
        self.end_task_run()
    
    def predict(self, image, label):
        # DNN forward pass
        size = (SAMPLE_SIZE, SAMPLE_SIZE)
        mean = 127.5
        factor = 1.0 / 127.5
        blob = cv2.dnn.blobFromImage(image, factor, size, mean, swapRB=False, crop=True)
        self.net.setInput(blob)
        outputs = self.net.forward()
        # prob = scipy.special.softmax(outputs)

        # Fill numeric output with class probabilities
        numeric_ouput = self.get_output(2)
        numeric_ouput.add_value_list(outputs.flatten().tolist(), label, self.class_names)

        # Get higher probability class and return its name
        class_index = np.argmax(outputs)
        return self.class_names[class_index]


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class EmotionFerPlusFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_emotion_fer_plus"
        self.info.short_description = "Facial emotion recognition using DNN trained from crowd-sourced label distribution."
        # relative path -> as displayed in Imageez application process tree
        self.info.path = "Plugins/Python/Face"
        self.info.version = "1.2.0"
        self.info.icon_path = "icon/icon.png"
        self.info.authors = "Emad Barsoum, Cha Zhang, Cristian Canton Ferrer and Zhengyou Zhang"
        self.info.article = "Training Deep Networks for Facial Expression Recognitionwith Crowd-Sourced Label Distribution"
        self.info.journal = "ACM ICMI"
        self.info.year = 2016
        self.info.license = "MIT License"
        self.info.documentation_link = "https://arxiv.org/pdf/1608.01041.pdf"
        self.info.repository = "https://github.com/Ikomia-hub/infer_emotion_fer_plus"
        self.info.original_repository = "https://github.com/microsoft/FERPlus"
        self.info.keywords = "face,expression,emotion,dnn"

    def create(self, param=None):
        # Create process object
        return EmotionFerPlus(self.info.name, param)
