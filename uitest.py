import dearpygui.dearpygui as dpg
import dearpygui.demo as demo

dpg.create_context()
dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.configure_app(docking=True, docking_space=True)
dpg.setup_dearpygui()

demo.show_demo()

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()