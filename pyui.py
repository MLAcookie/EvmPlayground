import dearpygui.dearpygui as dpg
import cv2
import evm_preview as evmp
import numpy as np
import terminal as term
import evm_motion as evmm


class PictruePreview:
    def RegComponent(self, name: str) -> int:
        self.__components[name] = dpg.generate_uuid()
        return self.__components[name]

    def GetComponentValue(self, name: str) -> any:
        return dpg.get_value(self.__components[name])

    def SetComponentValue(self, name: str, value: any) -> None:
        dpg.set_value(self.__components[name], value)

    def ConfigComponent(self, name: str, **kwargs: any) -> None:
        dpg.configure_item(self.__components[name], **kwargs)

    def __IndexChangeCallback(self, sender, data):
        self.SetComponentValue("Texture", self.imgs[data].ravel())

    def __init__(
        self,
        picSize: tuple[int, int],
        imgList: list[np.ndarray],
        windowName: str = "Preview",
        W: int = 500,
        H: int = 600,
    ) -> None:
        self.__components = {}
        self.imgs = imgList
        self.picSize = picSize
        if len(imgList) == 0:
            return
        data = imgList[0].ravel()
        with dpg.texture_registry():
            dpg.add_raw_texture(
                picSize[0],
                picSize[1],
                data,
                tag=self.RegComponent("Texture"),
                format=dpg.mvFormat_Float_rgb,
            )
        with dpg.window(label=windowName, width=W, height=H, min_size=[W, H + 40]):
            with dpg.plot(width=-1, height=H - 20, equal_aspects=True, crosshairs=True):
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    no_gridlines=True,
                    no_tick_labels=True,
                    no_tick_marks=True,
                )
                with dpg.plot_axis(
                    dpg.mvYAxis,
                    invert=True,
                    no_gridlines=True,
                    no_tick_labels=True,
                    no_tick_marks=True,
                ):
                    dpg.add_image_series(
                        self.__components["Texture"],
                        [0, self.picSize[1]],
                        [self.picSize[0], 0],
                    )
            dpg.add_slider_int(
                width=-1,
                height=20,
                format="Frame: %d",
                min_value=0,
                max_value=len(imgList) - 1,
                callback=self.__IndexChangeCallback,
            )


class EvmWindow:
    def RegComponent(self, name: str) -> int:
        self.__components[name] = dpg.generate_uuid()
        return self.__components[name]

    def GetComponentValue(self, name: str) -> any:
        return dpg.get_value(self.__components[name])

    def SetComponentValue(self, name: str, value: any) -> None:
        dpg.set_value(self.__components[name], value)

    def ConfigComponent(self, name: str, **kwargs: any) -> None:
        dpg.configure_item(self.__components[name], **kwargs)

    def ShowComponent(self, *name: str) -> None:
        for n in name:
            dpg.show_item(self.__components[n])

    def HideComponent(self, *name: str) -> None:
        for n in name:
            dpg.hide_item(self.__components[n])

    def __InitFileDialog(self) -> None:

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.__EVMFileDialogConfirmCallback,
            tag=self.RegComponent("FindVideo_FileDialog"),
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".mp4", color=(0, 255, 0, 255), custom_text="[mp4]")
            dpg.add_file_extension(".avi", color=(0, 255, 0, 255), custom_text="[avi]")

        dpg.add_file_dialog(
            directory_selector=True,
            show=False,
            callback=None,
            tag=self.RegComponent("SetPath_FileDialog"),
            width=700,
            height=400,
        )

    def __StartLoading(self):
        self.term = term.Terminal()

    def __EndLoading(self):
        self.term.Close()

    # region Callbacks

    def __PrintValueCallback(self, sender, data):
        print(f"Sender: {sender}, AppData: {data}")

    def __EVMFileDialogConfirmCallback(self, sender, data):
        self.path = data["file_path_name"]
        self.SetComponentValue("InputPath_Text", data["file_path_name"])

        videoCap = cv2.VideoCapture(data["file_path_name"])
        temp: int = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        self.fps = videoCap.get(cv2.CAP_PROP_FPS)
        self.frameSize = (
            int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        pixMax = max(self.frameSize[0], self.frameSize[1])
        x = 512 / pixMax
        self.importSize = (round(self.frameSize[0] * x), round(self.frameSize[1] * x))
        self.SetComponentValue("ImportSize_Text", f"Import Size: {self.importSize}")
        self.SetComponentValue("Resize_DragFloat", x)

        self.SetComponentValue("FPS_Text", f"FPS: {self.fps}")
        self.SetComponentValue("FrameCount_Text", f"Frame Count: {temp}")
        self.SetComponentValue("FrameEnd_SliderInt", temp)
        self.SetComponentValue("FrameSize_Text", f"Frame Size: {self.frameSize}")

        self.ConfigComponent("MainWindow", label="EVM: " + data["file_path_name"])
        self.ConfigComponent("FrameStart_SliderInt", max_value=temp)
        self.ConfigComponent("FrameEnd_SliderInt", max_value=temp)
        self.ConfigComponent("PixelLocateX_SliderInt", max_value=self.importSize[0] - 1)
        self.ConfigComponent("PixelLocateY_SliderInt", max_value=self.importSize[1] - 1)

        self.ShowComponent(
            "InputPath_Text", "FPS_Text", "FrameCount_Text", "FrameSize_Text", "ImportSize_Text"
        )

    def __EVMShowSignalButtonCallback(self):
        tlist = [t * 1000 / self.fps for t in range(len(self.frames))]
        temp = []
        tempFFT = []
        for _ in range(3):
            temp.append([])
            tempFFT.append([])
        for i in range(len(self.frames)):
            for j in range(0, 3):
                temp[j].append(
                    self.imgPyramids[3][
                        i,
                        self.GetComponentValue("PixelLocateY_SliderInt") >> 3,
                        self.GetComponentValue("PixelLocateX_SliderInt") >> 3,
                        j,
                    ]
                )
        for i in range(3):
            t = np.fft.fft(temp[i])
            tempFFT[i] = np.abs(t)
            self.SetComponentValue(f"{self.channels[i]}Signal_LineSeries", [tlist, temp[i]])
            self.SetComponentValue(f"{self.channels[i]}FFTSignal_LineSeries", [tlist, tempFFT[i]])
            dpg.fit_axis_data(self.__components[f"{self.channels[i]}Y_Axis"])
            dpg.fit_axis_data(self.__components[f"{self.channels[i]}X_Axis"])
            dpg.fit_axis_data(self.__components[f"{self.channels[i]}FFT_X_Axis"])
            dpg.fit_axis_data(self.__components[f"{self.channels[i]}FFT_Y_Axis"])

    def __EVMFilteredCalculateButtonCallback(self):
        self.__StartLoading()
        if self.GetComponentValue("FilterTypeTemp_RadioButton") == "ideal":
            self.filteredPyramids = evmp.idealFilterForVideoPyr(
                self.imgPyramids,
                self.GetComponentValue("FreqLowTemp_DragFloat"),
                self.GetComponentValue("FreqHighTemp_DragFloat"),
                self.fps,
            )
        else:
            self.filteredPyramids = evmp.buttFilterForVideoPyr(
                self.imgPyramids,
                self.GetComponentValue("FreqLowTemp_DragFloat"),
                self.GetComponentValue("FreqHighTemp_DragFloat"),
                self.fps,
            )
        tlist = [t * 1000 / self.fps for t in range(len(self.frames))]
        temp = []
        for _ in range(3):
            temp.append([])
        for i in range(len(self.frames)):
            for j in range(3):
                temp[j].append(
                    self.filteredPyramids[3][
                        i,
                        self.GetComponentValue("PixelLocateY_SliderInt") >> 3,
                        self.GetComponentValue("PixelLocateX_SliderInt") >> 3,
                        j,
                    ]
                )
        for i in range(3):
            self.SetComponentValue(f"F{self.channels[i]}Signal_LineSeries", [tlist, temp[i]])
            dpg.fit_axis_data(self.__components[f"F{self.channels[i]}Y_Axis"])
            dpg.fit_axis_data(self.__components[f"F{self.channels[i]}X_Axis"])

        self.__EndLoading()

    def __EVMFinalPreviewCallback(self):
        self.__StartLoading()
        temp = []
        temp = evmp.emvCoreColor(
            frames=self.frames,
            fps=self.fps,
            maxLevel=4,
            freqLow=self.GetComponentValue("FreqLowPreview_DragFloat"),
            freqHigh=self.GetComponentValue("FreqHighPreview_DragFloat"),
            alpha=self.GetComponentValue("AlphaPreview_DragFloat"),
            chromAttenuation=self.GetComponentValue("ChromAttenuation_DragFloat"),
            method=self.GetComponentValue("FilterType_RadioButton"),
        )

        out = []
        for f in temp:
            out.append(evmp.yiq2rgbFloat(f))
        PictruePreview(self.importSize, out)
        self.__EndLoading()

    # endregion Callbacks

    def __init__(self, W: int = 600, H: int = 800) -> None:

        self.__components = {}
        self.fps = 0
        self.importSize = (0, 0)
        self.frameSize = (0, 0)
        self.frames = []
        self.imgPyramids = []
        self.filteredPyramids = []
        self.channels = ["Y", "I", "Q"]
        self.path = ""

        self.__InitFileDialog()
        with dpg.window(
            tag=self.RegComponent("MainWindow"),
            label="New EVM Session",
            width=W,
            height=H,
        ):
            with dpg.collapsing_header(label="Load", default_open=True):
                dpg.add_text(
                    tag=self.RegComponent("InputPath_Text"),
                    default_value="Path: Null",
                    show=False,
                )
                dpg.add_text(
                    tag=self.RegComponent("FPS_Text"),
                    default_value="FPS: Path Null",
                    show=False,
                )
                dpg.add_text(
                    tag=self.RegComponent("FrameCount_Text"),
                    default_value="Frame Count: Path Null",
                    show=False,
                )
                dpg.add_text(
                    tag=self.RegComponent("FrameSize_Text"),
                    label="Frame Size: Path Null",
                    show=False,
                )
                with dpg.table(header_row=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                        dpg.add_slider_int(
                            width=-1,
                            max_value=0,
                            min_value=0,
                            default_value=0,
                            format="Frame Start: %d",
                            callback=self.__PrintValueCallback,
                            tag=self.RegComponent("FrameStart_SliderInt"),
                        )
                        dpg.add_slider_int(
                            width=-1,
                            max_value=0,
                            min_value=0,
                            default_value=0,
                            format="Frame End: %d",
                            callback=self.__PrintValueCallback,
                            tag=self.RegComponent("FrameEnd_SliderInt"),
                        )
                # region Resize
                with dpg.table(header_row=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():

                        def Resize_DragFloat_Callback(sender, data):
                            self.importSize = (
                                round(self.frameSize[0] * data),
                                round(self.frameSize[1] * data),
                            )
                            self.SetComponentValue("ImportSize_Text", f"Import Size: {self.importSize}")

                        dpg.add_slider_float(
                            width=-1,
                            tag=self.RegComponent("Resize_DragFloat"),
                            default_value=0,
                            min_value=0.01,
                            max_value=1,
                            format="Resize: %.2f",
                            callback=Resize_DragFloat_Callback,
                        )
                        dpg.add_text(
                            tag=self.RegComponent("ImportSize_Text"),
                            label="Impoet Size: Path Null",
                            show=False,
                        )
                # region Import Load Pewview
                with dpg.group(horizontal=True):

                    def PreviewButtonCallback():
                        self.__StartLoading()
                        term.Terminal.Clear()
                        term.Terminal.Println("Loading...")
                        temp = []
                        for f in self.frames:
                            temp.append(evmp.yiq2rgbFloat(f))
                        self.__EndLoading()
                        PictruePreview(self.importSize, temp)

                    def LoadButtonCallback():
                        if self.GetComponentValue("InputPath_Text") == "Path: Null":
                            return
                        self.__StartLoading()
                        _, self.frames = evmp.getVideoFrames(
                            self.GetComponentValue("InputPath_Text"),
                            self.GetComponentValue("FrameStart_SliderInt"),
                            self.GetComponentValue("FrameEnd_SliderInt"),
                            self.GetComponentValue("Resize_DragFloat"),
                        )
                        self.imgPyramids = evmp.buildVideoLapPyr(self.frames, 4)
                        self.__EndLoading()

                    dpg.add_button(
                        label="Import", callback=lambda: self.ShowComponent("FindVideo_FileDialog")
                    ),
                    dpg.add_button(label="Load", callback=LoadButtonCallback)
                    dpg.add_button(
                        label="Preview",
                        callback=PreviewButtonCallback,
                    )
            with dpg.collapsing_header(label="Analyse"):
                with dpg.table(header_row=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                        dpg.add_slider_int(
                            width=-1,
                            min_value=0,
                            max_value=0,
                            default_value=0,
                            format="X: %d",
                            tag=self.RegComponent("PixelLocateX_SliderInt"),
                            callback=self.__PrintValueCallback,
                        )
                        dpg.add_slider_int(
                            width=-1,
                            min_value=0,
                            max_value=0,
                            default_value=0,
                            format="Y: %d",
                            tag=self.RegComponent("PixelLocateY_SliderInt"),
                            callback=self.__PrintValueCallback,
                        )
                        with dpg.group(horizontal=True):
                            dpg.add_button(
                                label="Show Signal",
                                callback=self.__EVMShowSignalButtonCallback,
                            )

                            def IsOriginCallback(sender, data):
                                if data:
                                    self.ShowComponent("YIQSignal_Grid")
                                    self.HideComponent("YIQFFTSignal_Grid")
                                else:
                                    self.ShowComponent("YIQFFTSignal_Grid")
                                    self.HideComponent("YIQSignal_Grid")

                            dpg.add_checkbox(
                                label="Origin?",
                                callback=IsOriginCallback,
                            )
                # region SignalTabel
                with dpg.table(
                    tag=self.RegComponent("YIQSignal_Grid"),
                    header_row=False,
                    show=False,
                ):
                    for _ in range(3):
                        dpg.add_table_column()
                    with dpg.table_row():
                        for i in range(3):
                            with dpg.plot(label=self.channels[i], width=-1):
                                dpg.add_plot_axis(
                                    dpg.mvXAxis,
                                    label="Time(ms)",
                                    tag=self.RegComponent(f"{self.channels[i]}X_Axis"),
                                )
                                dpg.add_plot_axis(
                                    dpg.mvYAxis,
                                    label="PixelValue",
                                    tag=self.RegComponent(f"{self.channels[i]}Y_Axis"),
                                )
                                dpg.add_line_series(
                                    x=[],
                                    y=[],
                                    parent=self.__components[f"{self.channels[i]}Y_Axis"],
                                    tag=self.RegComponent(f"{self.channels[i]}Signal_LineSeries"),
                                )
                with dpg.table(
                    tag=self.RegComponent("YIQFFTSignal_Grid"),
                    header_row=False,
                ):
                    for _ in range(3):
                        dpg.add_table_column()
                    with dpg.table_row():
                        for i in range(3):
                            with dpg.plot(label=f"{self.channels[i]}_FFT", width=-1):
                                dpg.add_plot_axis(
                                    dpg.mvXAxis,
                                    label="Freq(hz)",
                                    log_scale=True,
                                    tag=self.RegComponent(f"{self.channels[i]}FFT_X_Axis"),
                                )
                                dpg.add_plot_axis(
                                    dpg.mvYAxis,
                                    label="Value",
                                    tag=self.RegComponent(f"{self.channels[i]}FFT_Y_Axis"),
                                )
                                dpg.add_line_series(
                                    x=[],
                                    y=[],
                                    parent=self.__components[f"{self.channels[i]}FFT_Y_Axis"],
                                    tag=self.RegComponent(f"{self.channels[i]}FFTSignal_LineSeries"),
                                )
                with dpg.table(header_row=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():

                        def FilterTypeTemp_RadioButton_Callback(sender, data):
                            self.SetComponentValue("FilterType_RadioButton", data)
                            if data == "butt":
                                self.ConfigComponent("FreqLowTemp_DragFloat", max_value=self.fps / 2)
                                self.ConfigComponent("FreqHighTemp_DragFloat", max_value=self.fps / 2)
                                self.ConfigComponent("FreqLowPreview_DragFloat", max_value=self.fps / 2)
                                self.ConfigComponent("FreqHighPreview_DragFloat", max_value=self.fps / 2)
                            else:
                                self.ConfigComponent("FreqLowTemp_DragFloat", max_value=9999)
                                self.ConfigComponent("FreqHighTemp_DragFloat", max_value=9999)
                                self.ConfigComponent("FreqLowPreview_DragFloat", max_value=9999)
                                self.ConfigComponent("FreqHighPreview_DragFloat", max_value=9999)

                        dpg.add_drag_float(
                            min_value=0,
                            max_value=9999,
                            width=-1,
                            format="Freq Low: %.3f hz",
                            tag=self.RegComponent("FreqLowTemp_DragFloat"),
                            callback=lambda s, d: self.SetComponentValue("FreqLowPreview_DragFloat", d),
                        )
                        dpg.add_drag_float(
                            min_value=0,
                            max_value=9999,
                            width=-1,
                            format="Freq High: %.3f hz",
                            tag=self.RegComponent("FreqHighTemp_DragFloat"),
                            callback=lambda s, d: self.SetComponentValue("FreqHighPreview_DragFloat", d),
                        )
                        with dpg.group(horizontal=True):
                            dpg.add_radio_button(
                                ["ideal", "butt"],
                                default_value="ideal",
                                horizontal=True,
                                tag=self.RegComponent("FilterTypeTemp_RadioButton"),
                                callback=FilterTypeTemp_RadioButton_Callback,
                            )
                            dpg.add_button(
                                label="Calculate",
                                callback=self.__EVMFilteredCalculateButtonCallback,
                            )
                # region FilteredTable
                with dpg.table(
                    tag=self.RegComponent("YIQFilteredSignal_Grid"),
                    header_row=False,
                ):
                    for _ in range(3):
                        dpg.add_table_column()
                    with dpg.table_row():
                        for i in range(3):
                            with dpg.plot(label=f"{self.channels[i]}_Filtered", width=-1):
                                dpg.add_plot_axis(
                                    dpg.mvXAxis,
                                    label="Time(ms)",
                                    tag=self.RegComponent(f"F{self.channels[i]}X_Axis"),
                                )
                                dpg.add_plot_axis(
                                    dpg.mvYAxis,
                                    label="PixelValue",
                                    tag=self.RegComponent(f"F{self.channels[i]}Y_Axis"),
                                )
                                dpg.add_line_series(
                                    x=[],
                                    y=[],
                                    parent=self.__components[f"F{self.channels[i]}Y_Axis"],
                                    tag=self.RegComponent(f"F{self.channels[i]}Signal_LineSeries"),
                                )

            with dpg.collapsing_header(label="Preview"):
                with dpg.table(header_row=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():

                        def FilterType_RadioButton_Callback(sender, data):
                            if data == "butt":
                                self.ConfigComponent("FreqLowPreview_DragFloat", max_value=self.fps / 2)
                                self.ConfigComponent("FreqHighPreview_DragFloat", max_value=self.fps / 2)
                            else:
                                self.ConfigComponent("FreqLowPreview_DragFloat", max_value=9999)
                                self.ConfigComponent("FreqHighPreview_DragFloat", max_value=9999)

                        dpg.add_drag_float(
                            width=-1,
                            min_value=0,
                            max_value=9999,
                            format="Freq Low: %.3f hz",
                            default_value=0,
                            callback=self.__PrintValueCallback,
                            tag=self.RegComponent("FreqLowPreview_DragFloat"),
                        )
                        dpg.add_drag_float(
                            width=-1,
                            min_value=0,
                            max_value=9999,
                            format="Freq High: %.3f hz",
                            default_value=0,
                            callback=self.__PrintValueCallback,
                            tag=self.RegComponent("FreqHighPreview_DragFloat"),
                        )
                        dpg.add_radio_button(
                            ["ideal", "butt"],
                            default_value="ideal",
                            horizontal=True,
                            tag=self.RegComponent("FilterType_RadioButton"),
                            callback=FilterType_RadioButton_Callback,
                        )
                    with dpg.table_row():
                        dpg.add_drag_float(
                            width=-1,
                            min_value=0,
                            max_value=9999,
                            tag=self.RegComponent("AlphaPreview_DragFloat"),
                            format="Preview Alpha: %.2f",
                            default_value=15,
                            callback=self.__PrintValueCallback,
                        )
                        dpg.add_drag_float(
                            width=-1,
                            min_value=-100,
                            max_value=100,
                            tag=self.RegComponent("ChromAttenuation_DragFloat"),
                            format="Chrom Attenuation: %.2f",
                            default_value=0,
                            callback=self.__PrintValueCallback,
                        )
                        dpg.add_button(label="Preview", callback=self.__EVMFinalPreviewCallback)
            with dpg.collapsing_header(label="Export"):
                with dpg.table(header_row=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                        dpg.add_drag_float(
                            width=-1,
                            min_value=0,
                            max_value=9999,
                            format="Freq Low: %.3f hz",
                            default_value=0,
                            callback=self.__PrintValueCallback,
                            tag=self.RegComponent("FreqLow_DragFloat"),
                        )
                        dpg.add_drag_float(
                            width=-1,
                            min_value=0,
                            max_value=9999,
                            format="Freq High: %.3f hz",
                            default_value=0,
                            callback=self.__PrintValueCallback,
                            tag=self.RegComponent("FreqHigh_DragFloat"),
                        )
                        dpg.add_drag_float(
                            width=-1,
                            min_value=0,
                            max_value=9999,
                            format="Alpha: %.2f",
                            default_value=10,
                            callback=self.__PrintValueCallback,
                            tag=self.RegComponent("Alpha_DragFloat"),
                        )
                    with dpg.table_row():
                        dpg.add_drag_float(
                            width=-1,
                            min_value=0,
                            max_value=9999,
                            format="Sigma: %.2f",
                            default_value=3,
                            callback=self.__PrintValueCallback,
                            tag=self.RegComponent("Sigma_DragFloat"),
                        )
                        dpg.add_drag_float(
                            width=-1,
                            min_value=0,
                            max_value=1,
                            format="Scale: %.2f",
                            default_value=0.5,
                            callback=self.__PrintValueCallback,
                            tag=self.RegComponent("Scale_DragFloat"),
                        )

                        def ExportButtonCallback():
                            self.__StartLoading()
                            evmm.phase_amplify_to_file(
                                self.path,
                                self.GetComponentValue("Alpha_DragFloat"),
                                self.GetComponentValue("FreqLow_DragFloat"),
                                self.GetComponentValue("FreqHigh_DragFloat"),
                                self.fps,
                                "./result",
                                sigma=self.GetComponentValue("Sigma_DragFloat"),
                                pyrtype="octave",
                                scalevideo=self.GetComponentValue("Scale_DragFloat"),
                            )
                            self.__EndLoading()

                        dpg.add_button(label="Export", callback=ExportButtonCallback)


if __name__ == "__main__":
    dpg.create_context()
    dpg.create_viewport(title="Test", width=800, height=600)
    dpg.configure_app(docking=True, docking_space=True)

    EvmWindow()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()

    dpg.destroy_context()
