import dearpygui.dearpygui as dpg
import cv2
import evm
import numpy as np


class Imgui:
    # region GlobalCallbacks

    def __PrintValueCallback(self, sender, data):
        print(f"Sender: {sender}, AppData: {data}")

    def __CloseViewpointCallback(self):
        dpg.destroy_context()

    # endregion

    # region WelcomeWindow

    def __ShowWelcomeWindow(self):
        with dpg.window(label="Welcome"):
            dpg.add_text("Welcome to This Tool Box\n")

    # endregino

    # region EVMConfigWindow

    evmVideoImportFlag: bool = False
    frames: list[np.ndarray] = []
    fps: int = 0
    imgPyramids = []
    filteredPyramids = []

    def __EVMFileDialogConfirmCallback(self, sender, data):
        print(f"Sender: {sender}, AppData: {data}")
        dpg.set_value("InputPath", data["file_path_name"])
        videoCap = cv2.VideoCapture(data["file_path_name"])
        temp: int = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        dpg.configure_item("FrameRange", max_value=temp)
        dpg.set_value("FrameRange", (0, temp, 0, 0))

    def __EVMLoadVideoCallback(self, sender, data):
        if dpg.get_value("InputPath") == "Path Null":
            return

        self.frames = []
        self.fps = 0
        self.fps, self.frames = evm.getVideoFrames(
            dpg.get_value("InputPath"),
            dpg.get_value("FrameRange")[0],
            dpg.get_value("FrameRange")[1],
        )
        self.imgPyramids = evm.buildVideoLapPyr(self.frames, dpg.get_value("Maxlevel"))
        print("Done")

    def __EVMShowSignalCallback(self):
        tlist = [
            t * 1000 / self.fps
            for t in range(self.imgPyramids[dpg.get_value("Maxlevel") - 1].shape[0])
        ]
        temp = []
        for i in range(0, 3):
            temp.append([])
        for i in range(len(self.frames)):
            for j in range(0, 3):
                temp[j].append(
                    self.imgPyramids[0][
                        i,
                        dpg.get_value("PixelLocate")[1],
                        dpg.get_value("PixelLocate")[0],
                        j,
                    ]
                )
        dpg.set_value("YSignal", [tlist, temp[0]])
        dpg.set_value("ISignal", [tlist, temp[1]])
        dpg.set_value("QSignal", [tlist, temp[2]])
        dpg.fit_axis_data("YYAxis")
        dpg.fit_axis_data("IYAxis")
        dpg.fit_axis_data("QYAxis")
        dpg.fit_axis_data("YXAxis")
        dpg.fit_axis_data("IXAxis")
        dpg.fit_axis_data("QXAxis")

        dpg.show_item("YIQSignal")

    def __EVMShowFilteredSignalCallback(self):
        self.filteredPyramids = evm.idealFilterForVideoPyr(
            self.imgPyramids,
            dpg.get_value("FrequencyRangeTemp")[0],
            dpg.get_value("FrequencyRangeTemp")[1],
            self.fps,
        )
        tlist = [
            t * 1000 / self.fps
            for t in range(self.imgPyramids[dpg.get_value("Maxlevel") - 1].shape[0])
        ]
        temp = []
        for i in range(0, 3):
            temp.append([])
        for i in range(len(self.frames)):
            for j in range(0, 3):
                temp[j].append(
                    self.filteredPyramids[0][
                        i,
                        dpg.get_value("PixelLocate")[1],
                        dpg.get_value("PixelLocate")[0],
                        j,
                    ]
                )
        dpg.set_value("FYSignal", [tlist, temp[0]])
        dpg.set_value("FISignal", [tlist, temp[1]])
        dpg.set_value("FQSignal", [tlist, temp[2]])
        dpg.fit_axis_data("FYYAxis")
        dpg.fit_axis_data("FIYAxis")
        dpg.fit_axis_data("FQYAxis")
        dpg.fit_axis_data("FYXAxis")
        dpg.fit_axis_data("FIXAxis")
        dpg.fit_axis_data("FQXAxis")

        dpg.show_item("YIQFilteredSignal")

    def __ShowEVMConfigWindow(self):
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.__EVMFileDialogConfirmCallback,
            tag="MP4FileDialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".mp4", color=(0, 255, 0, 255), custom_text="[mp4]")

        with dpg.window(label="EVM", width=500, height=400):
            with dpg.collapsing_header(label="Load", default_open=True):
                dpg.add_button(
                    label="Import", callback=lambda: dpg.show_item("MP4FileDialog")
                )
                dpg.add_same_line()
                dpg.add_text(tag="InputPath", default_value="Path Null")
                dpg.add_slider_intx(
                    size=2,
                    max_value=0,
                    min_value=0,
                    default_value=(0, 0, 0, 0),
                    label="FrameRange",
                    callback=self.__PrintValueCallback,
                    tag="FrameRange",
                )
                dpg.add_input_int(
                    tag="Maxlevel",
                    default_value=4,
                    label="Maxlevel",
                )
                dpg.add_button(label="Load", callback=self.__EVMLoadVideoCallback)
            with dpg.collapsing_header(label="Analyse"):
                dpg.add_drag_intx(
                    size=2,
                    min_value=0,
                    default_value=(240, 180, 0, 0),
                    label="PixelLocate",
                    tag="PixelLocate",
                    callback=self.__PrintValueCallback,
                )
                dpg.add_button(label="ShowPlot", callback=self.__EVMShowSignalCallback)
                with dpg.table(tag="YIQSignal", header_row=False, show=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                        with dpg.plot(label="Y", width=-1):
                            dpg.add_plot_axis(
                                dpg.mvXAxis, label="Time(ms)", tag="YXAxis"
                            )
                            dpg.add_plot_axis(
                                dpg.mvYAxis, label="PixelValue", tag="YYAxis"
                            )
                            dpg.add_line_series(
                                x=[],
                                y=[],
                                parent="YYAxis",
                                tag="YSignal",
                            )
                        with dpg.plot(label="I", width=-1):
                            dpg.add_plot_axis(
                                dpg.mvXAxis, label="Time(ms)", tag="IXAxis"
                            )
                            dpg.add_plot_axis(
                                dpg.mvYAxis, label="PixelValue", tag="IYAxis"
                            )
                            dpg.add_line_series(
                                x=[],
                                y=[],
                                parent="IYAxis",
                                tag="ISignal",
                            )
                        with dpg.plot(label="Q", width=-1):
                            dpg.add_plot_axis(
                                dpg.mvXAxis, label="Time(ms)", tag="QXAxis"
                            )
                            dpg.add_plot_axis(
                                dpg.mvYAxis, label="PixelValue", tag="QYAxis"
                            )
                            dpg.add_line_series(
                                x=[],
                                y=[],
                                parent="QYAxis",
                                tag="QSignal",
                            )
                dpg.add_slider_floatx(
                    size=2,
                    min_value=0,
                    default_value=(0, 0, 0, 0),
                    label="FrequencyRange",
                    callback=self.__PrintValueCallback,
                    tag="FrequencyRangeTemp",
                )
                dpg.add_button(
                    label="ShowFilteredSignal",
                    callback=self.__EVMShowFilteredSignalCallback,
                )
                with dpg.table(tag="YIQFilteredSignal", header_row=False, show=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                        with dpg.plot(label="Y", width=-1):
                            dpg.add_plot_axis(
                                dpg.mvXAxis, label="Time(ms)", tag="FYXAxis"
                            )
                            dpg.add_plot_axis(
                                dpg.mvYAxis, label="PixelValue", tag="FYYAxis"
                            )
                            dpg.add_line_series(
                                x=[],
                                y=[],
                                parent="FYYAxis",
                                tag="FYSignal",
                            )
                        with dpg.plot(label="I", width=-1):
                            dpg.add_plot_axis(
                                dpg.mvXAxis, label="Time(ms)", tag="FIXAxis"
                            )
                            dpg.add_plot_axis(
                                dpg.mvYAxis, label="PixelValue", tag="FIYAxis"
                            )
                            dpg.add_line_series(
                                x=[],
                                y=[],
                                parent="FIYAxis",
                                tag="FISignal",
                            )
                        with dpg.plot(label="Q", width=-1):
                            dpg.add_plot_axis(
                                dpg.mvXAxis, label="Time(ms)", tag="FQXAxis"
                            )
                            dpg.add_plot_axis(
                                dpg.mvYAxis, label="PixelValue", tag="FQYAxis"
                            )
                            dpg.add_line_series(
                                x=[],
                                y=[],
                                parent="FQYAxis",
                                tag="FQSignal",
                            )

            with dpg.collapsing_header(label="Export"):
                dpg.add_slider_floatx(
                    size=2,
                    min_value=0,
                    default_value=(0, 0, 0, 0),
                    label="FrequencyRange",
                    callback=self.__PrintValueCallback,
                    tag="FrequencyRange",
                )
                dpg.add_input_float(
                    tag="AmplificationRate",
                    label="AmplificationRate",
                    default_value=150,
                    callback=self.__PrintValueCallback,
                )

    # endregion

    def __SetMenuItem(self):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="Start"):
                dpg.add_menu_item(label="Welcome", callback=self.__ShowWelcomeWindow)
                dpg.add_menu_item(label="EVM", callback=self.__ShowEVMConfigWindow)
                dpg.add_menu_item(label="Exit", callback=self.__CloseViewpointCallback)
            with dpg.menu(label="Tool"):
                dpg.add_menu_item(label="About Imgui", callback=dpg.show_about)
                dpg.add_menu_item(label="Performance", callback=dpg.show_metrics)

    def __init__(self, w: int = 800, h: int = 600) -> None:
        dpg.create_context()
        dpg.create_viewport(title="MotionEnhanceToolBox", width=w, height=h)
        dpg.configure_app(docking=True, docking_space=True)
        dpg.set_exit_callback(self.__CloseViewpointCallback)

        self.__SetMenuItem()
        self.__ShowEVMConfigWindow()

    # region Public

    def Show(self):
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()

    # endregion


if __name__ == "__main__":
    temp = Imgui()
    temp.Show()
