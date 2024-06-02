import dearpygui.dearpygui as dpg


class Terminal:
    text: str = ""
    lastTextlenth: int = -1
    textComponents: list[int] = []

    def __Sync():
        for t in Terminal.textComponents:
            dpg.set_value(t, Terminal.text)

    def Clear():
        Terminal.text = ""
        Terminal.__Sync()

    def Print(string: str):
        if len(Terminal.text) > 1000:
            Terminal.Clear()
        Terminal.text += string
        Terminal.lastTextlenth = len(string)
        Terminal.__Sync()

    def Println(string: str):
        Terminal.Print(string + "\n")

    def DelteLast():
        if Terminal.lastTextlenth == -1:
            return
        Terminal.text = Terminal.text[: -Terminal.lastTextlenth]
        Terminal.__Sync()
        Terminal.lastTextlenth = -1

    def Close(self):
        Terminal.textComponents.remove(self.textComponent)
        dpg.delete_item(self.window)

    def __init__(self, warp: int = 800):
        self.textComponent: int = dpg.generate_uuid()
        Terminal.textComponents.append(self.textComponent)
        self.window: int = dpg.generate_uuid()

        def OnWindowClose():
            Terminal.textComponents.remove(self.textComponent)

        with dpg.window(label="Output", tag=self.window, width=400, height=400, on_close=OnWindowClose):
            dpg.add_text(tag=self.textComponent, default_value=Terminal.text, wrap=warp)


# if __name__ == "__main__":
#     dpg.create_context()
#     dpg.create_viewport(title="Test", width=800, height=600)
#     dpg.configure_app(docking=True, docking_space=True)

#     terminal = Terminal()
#     Terminal.Println("Hello World.")
#     terminal2 = Terminal()
#     Terminal.Println("Hello World2")
#     Terminal.DelteLast()
#     Terminal.Println("Hello World3")
#     terminal2.Close()
#     terminal3 = Terminal()

#     dpg.setup_dearpygui()
#     dpg.show_viewport()
#     dpg.start_dearpygui()

#     dpg.destroy_context()
