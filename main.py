import dearpygui.dearpygui as dpg
import terminal as term
import pyui as ui


class Imgui:
    # region Callbacks

    def __CloseViewpointCallback(self):
        dpg.destroy_context()

    def __ShowWelcomeWindow(self):
        with dpg.window(label="Welcome", width=800, height=600):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=200):
                    pass
                with dpg.child_window(width=-1):
                    dpg.add_text()

    # endregion Callbacks

    def __SetMenuItem(self):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="Start"):
                dpg.add_menu_item(label="Welcome", callback=self.__ShowWelcomeWindow)
                dpg.add_menu_item(label="New EVM Session", callback=lambda: ui.EvmWindow())
                dpg.add_menu_item(label="Output Info", callback=lambda: term.Terminal())
                dpg.add_menu_item(label="Exit", callback=self.__CloseViewpointCallback)
            with dpg.menu(label="Tool"):
                dpg.add_menu_item(label="About Imgui", callback=dpg.show_about)
                dpg.add_menu_item(label="Performance", callback=dpg.show_metrics)

    def __init__(self, w: int = 1000, h: int = 800):
        dpg.create_context()
        dpg.create_viewport(title="MotionEnhanceToolBox", width=w, height=h)
        dpg.configure_app(docking=True, docking_space=True)
        dpg.set_exit_callback(self.__CloseViewpointCallback)

        self.__SetMenuItem()
        ui.EvmWindow()

    # region Public

    def Show(self):
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()

    # endregion Public


if __name__ == "__main__":
    temp = Imgui()
    temp.Show()
