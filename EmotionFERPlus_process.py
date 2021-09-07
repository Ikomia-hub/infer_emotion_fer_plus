from ikomia import core, dataprocess
import copy
import os
import cv2
import numpy as np

SAMPLE_SIZE = 64

# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class EmotionFERPlusProcessParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.update = False
        self.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        self.target = cv2.dnn.DNN_TARGET_CPU
        self.model_path = os.path.dirname(os.path.realpath(__file__)) + "/models/model.onnx"

    def setParamMap(self, param_map):
        # Set parameters values from Imageez application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(paramMap["windowSize"])
        pass

    def getParamMap(self):
        # Send parameters values to Imageez application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        # Example : paramMap["windowSize"] = str(self.windowSize)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class EmotionFERPlusProcess(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CNumericIO())

        # Network members
        self.net = None
        self.class_names = []

        # Create parameters class
        if param is None:
            self.setParam(EmotionFERPlusProcessParam())
        else:
            self.setParam(copy.deepcopy(param))

        # Load class names
        model_folder = os.path.dirname(os.path.realpath(__file__)) + "/models"
        with open(model_folder + "/class_names") as f:
            for row in f:
                self.class_names.append(row[:-1])

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Imageez application
        return 4

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Forward input image
        self.forwardInputImage(0, 0)

        # Get parameters :
        param = self.getParam()

        # Load the recognition model from disk
        if self.net is None or param.update == True:
            self.net = cv2.dnn.readNet(param.model_path)
            self.net.setPreferableBackend(param.backend)
            self.net.setPreferableTarget(param.target)
            param.update = False

        # Step progress bar:
        self.emitStepProgress()

        # Get input :
        input_img = self.getInput(0)
        input_graphics = self.getInput(1)

        # Init graphics output
        graphics_output = self.getOutput(1)
        graphics_output.setNewLayer("EmotionFerPlus")
        graphics_output.setImageIndex(0)

        # Init numeric output
        numeric_ouput = self.getOutput(2)
        numeric_ouput.clearData()
        numeric_ouput.setOutputType(dataprocess.NumericOutputType.TABLE)

        # Step progress bar:
        self.emitStepProgress()

        # Get image from input (numpy array):
        src_image = input_img.getImage()
        if src_image.ndim == 3:
            gray_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = src_image

        # Step progress bar:
        self.emitStepProgress()

        # Predict
        if input_graphics.isDataAvailable():
            index = 1
            items = input_graphics.getItems()

            for item in items:
                if item.getType() == core.GraphicsItem.RECTANGLE:
                    x = int(item.x)
                    y = int(item.y)
                    w = int(item.width)
                    h = int(item.height)
                    face_img = gray_image[y:y+h, x:x+w]
                    emotion = self.predict(face_img, "Face #"+ str(index))
                    graphics_output.addText(emotion, x + (0.05*w), y + (0.05*h))
                    index += 1   
        else:
            (h,w) = gray_image.shape
            emotion = self.predict(gray_image, "Full image")
            graphics_output.addText(emotion, 0.05*w, 0.05*h)     

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process       
        self.endTaskRun()
    
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
        numeric_ouput = self.getOutput(2)
        numeric_ouput.addValueList(outputs.flatten().tolist(), label, self.class_names)

        # Get higher probability class and return its name
        class_index = np.argmax(outputs)
        return self.class_names[class_index]


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class EmotionFERPlusProcessFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "EmotionFERPlus"
        self.info.shortDescription = "Facial emotion recognition using DNN trained from crowd-sourced label distribution."
        self.info.description = "Crowd  sourcing  has  become  a  widely  adopted  scheme  tocollect  ground  truth  labels.   However,  it  is  a  well-knownproblem that these labels can be very noisy.  In this paper,we  demonstrate  how  to  learn  a  deep  convolutional  neuralnetwork (DCNN) from noisy labels, using facial expressionrecognition  as  an  example.   More  specifically,  we  have  10taggers  to  label  each  input  image,  and  compare  four  dif-ferent  approaches  to  utilizing  the  multiple  labels:   major-ity voting, multi-label learning, probabilistic label drawing,and cross-entropy loss.  We show that the traditional major-ity voting scheme does not perform as well as the last twoapproaches  that  fully  leverage  the  label  distribution.   Anenhanced FER+ data set with multiple labels for each faceimage will also be shared with the research community."
        # relative path -> as displayed in Imageez application process tree
        self.info.path = "Plugins/Python/Face"
        self.info.version = "1.0.0"
        self.info.iconPath = "icon/icon.png"
        self.info.authors = "Emad Barsoum, Cha Zhang, Cristian Canton Ferrer and Zhengyou Zhang"
        self.info.article = "Training Deep Networks for Facial Expression Recognitionwith Crowd-Sourced Label Distribution"
        self.info.journal = "ACM ICMI"
        self.info.year = 2016
        self.info.license = "MIT License"
        self.info.documentationLink = "https://arxiv.org/pdf/1608.01041.pdf"
        self.info.repository = "https://github.com/microsoft/FERPlus"
        self.info.keywords = "face,expression,emotion,dnn"

    def create(self, param=None):
        # Create process object
        return EmotionFERPlusProcess(self.info.name, param)
