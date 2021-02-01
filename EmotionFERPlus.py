from ikomia import dataprocess
import EmotionFERPlus_process as processMod
import EmotionFERPlus_widget as widgetMod


#--------------------
#- Interface class to integrate the process with Imageez application
#- Inherits dataprocess.CPluginProcessInterface from Imageez API
#--------------------
class EmotionFERPlus(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        #Instantiate process object
        return processMod.EmotionFERPlusProcessFactory()

    def getWidgetFactory(self):
        #Instantiate associated widget object
        return widgetMod.EmotionFERPlusWidgetFactory()
