from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia Studio
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from infer_emotion_fer_plus.infer_emotion_fer_plus_process import EmotionFerPlusFactory
        # Instantiate process object
        return EmotionFerPlusFactory()

    def getWidgetFactory(self):
        from infer_emotion_fer_plus.infer_emotion_fer_plus_widget import EmotionFerPlusWidgetFactory
        # Instantiate associated widget object
        return EmotionFerPlusWidgetFactory()
