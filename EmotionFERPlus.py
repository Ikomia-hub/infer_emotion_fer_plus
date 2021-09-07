from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia Studio
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class EmotionFERPlus(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from EmotionFERPlus.EmotionFERPlus_process import EmotionFERPlusProcessFactory
        # Instantiate process object
        return EmotionFERPlusProcessFactory()

    def getWidgetFactory(self):
        from EmotionFERPlus.EmotionFERPlus_widget import EmotionFERPlusWidgetFactory
        # Instantiate associated widget object
        return EmotionFERPlusWidgetFactory()
