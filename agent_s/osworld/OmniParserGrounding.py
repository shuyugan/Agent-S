from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
import os
import base64
import io
from typing import Dict, List, Tuple
from ultralytics import YOLO
from PIL import Image
device = 'cuda'

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)

def agent_action(func):
    func.is_agent_action = True
    return func

class GroundingAgent:
    def __init__(self, semantic_info=True):

        weights_path = os.path.join(parent_path, 'weights')
        self.temp_screenshot_path = os.path.join(current_path, 'temp')
        self.cnt = 0
        os.makedirs(self.temp_screenshot_path, exist_ok=True)
        

        self.som_model = get_yolo_model(model_path=os.path.join(weights_path, 'icon_detect', 'best.pt'))
        self.som_model.to(device)
        self.caption_model = get_caption_model_processor(model_name="florence2", model_name_or_path=os.path.join(weights_path, 'icon_caption_florence'), device=device)
        self.box_threshold = 0.03
        # self.box_overlay_ratio = image.size[0] / 3200
        self.semantic_info = semantic_info
        self.info = {}
        
        self.index_out_of_range_flag = False
        self.app_setup_code = f"""import subprocess;
import difflib;
import pyautogui;
pyautogui.press('escape');
time.sleep(0.5);
output = subprocess.check_output(['wmctrl', '-lx']);
output = output.decode('utf-8').splitlines();
window_titles = [line.split(None, 4)[2] for line in output];
closest_matches = difflib.get_close_matches('APP_NAME', window_titles, n=1, cutoff=0.1);
if closest_matches:
    closest_match = closest_matches[0];
    for line in output:
        if closest_match in line:
            window_id = line.split()[0]
            break;
subprocess.run(['wmctrl', '-ia', window_id])
subprocess.run(['wmctrl', '-ir', window_id, '-b', 'add,maximized_vert,maximized_horz'])
"""
        self.notes = []
        self.clipboard = ""

    def parse_image(self, screenshot):

        with open(os.path.join(self.temp_screenshot_path, f"{self.cnt}.png"),
                      "wb") as _f:
                _f.write(screenshot)

        box_overlay_ratio = screenshot.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        image_path = os.path.join(self.temp_screenshot_path, f"{self.cnt}.png")

        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img=True, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=True)
        text, ocr_bbox = ocr_bbox_rslt

        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_path, self.som_model, BOX_TRESHOLD = self.box_threshold, output_coord_in_ratio=False, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.1, imgsz=640)

        som_screenshot = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        
        parsed_caption_lst = ["Box\tCaption"] 

        for i in range(len(parsed_content_list)):
             name = parsed_content_list[i].split(':')[0]
             caption = parsed_content_list[i].split(':')[1].strip()
             self.info[i]['caption'] = caption
             parsed_caption_lst.append(
                 f"{name}\t{caption}"
             )
        parsed_caption = "\n".join(parsed_caption_lst)

        for a in label_coordinates:
             ID = int(a)
             x, y, w, h = label_coordinates[a][0], label_coordinates[a][1], label_coordinates[a][2], label_coordinates[a][3]
             center_coordinates = (x + w / 2, y + h / 2)
             size = (w, h)
             self.info[ID]['location'] = center_coordinates
             self.info[ID]['size'] = size

        return som_screenshot, parsed_caption
     
    def find_element(self, element_id):
        try:
            selected_element = self.info[int(element_id)]
        except:
            print("The index of the selected element was out of range.")
            selected_element = self.info[0]
            self.index_out_of_range_flag = True
        return selected_element
    
    @agent_action
    def click(
        self,
        element_id: int,
        num_clicks: int = 1,
        button_type: str = "left",
        hold_keys: List = [],
    ):
        """Click on the element
        Args:
            element_id:int, ID of the element to click on
            num_clicks:int, number of times to click the element
            button_type:str, which mouse button to press can be "left", "middle", or "right"
            hold_keys:List, list of keys to hold while clicking
        """
        node = self.find_element(element_id)
        x, y = node['location']

        command = "import pyautogui; "

        # TODO: specified duration?
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"""import pyautogui; pyautogui.click({x}, {y}, clicks={num_clicks}, button={repr(button_type)}); """
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
        # Return pyautoguicode to click on the element
        return command

    @agent_action
    def switch_applications(self, app_code):
        """Switch to a different application that is already open
        Args:
            app_code:str the code name of the application to switch to from the provided list of open applications
        """
        return self.app_setup_code.replace("APP_NAME", app_code)

    @agent_action
    def type(
        self,
        element_id: int = None,
        text: str = '',
        overwrite: bool = False,
        enter: bool = False,
    ):
        """Type text into the element
        Args:
            element_id:int ID of the element to type into. If not provided, typing will start at the current cursor location.
            text:str the text to type
            overwrite:bool Assign it to True if the text should overwrite the existing text, otherwise assign it to False. Using this argument clears all text in an element.
            enter:bool Assign it to True if the enter key should be pressed after typing the text, otherwise assign it to False.
        """
        try:
            # Use the provided element_id or default to None
            node = self.find_element(element_id) if element_id is not None else None
        except:
            node = None

        if node is not None:
            # If a node is found, retrieve its coordinates and size
            # coordinates = eval(
            #     node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
            # )
            # sizes = eval(node.get("{{{:}}}size".format(component_ns), "(-1, -1)"))

            # # Calculate the center of the element
            # x = coordinates[0] + sizes[0] // 2
            # y = coordinates[1] + sizes[1] // 2
            x, y = node['location']

            # Start typing at the center of the element
            command = "import pyautogui; "
            command += f"pyautogui.click({x}, {y}); "

            if overwrite:
                command += (
                    f"pyautogui.hotkey('ctrl', 'a'); pyautogui.press('backspace'); "
                )

            command += f"pyautogui.write({repr(text)}); "

            if enter:
                command += "pyautogui.press('enter'); "
        else:
            # If no element is found, start typing at the current cursor location
            command = "import pyautogui; "

            if overwrite:
                command += (
                    f"pyautogui.hotkey('ctrl', 'a'); pyautogui.press('backspace'); "
                )

            command += f"pyautogui.write({repr(text)}); "

            if enter:
                command += "pyautogui.press('enter'); "

        return command

    @agent_action
    def save_to_knowledge(self, text: List[str]):
        """Save facts, elements, texts, etc. to a long-term knowledge bank for reuse during this task. Can be used for copy-pasting text, saving elements, etc.
        Args:
            text:List[str] the text to save to the knowledge
        """
        self.notes.extend(text)
        return """WAIT"""

    @agent_action
    def drag_and_drop(self, drag_from_id: int, drop_on_id: int, hold_keys: List = []):
        """Drag element1 and drop it on element2.
        Args:
            drag_from_id:int ID of element to drag
            drop_on_id:int ID of element to drop on
            hold_keys:List list of keys to hold while dragging
        """
        node1 = self.find_element(drag_from_id)
        node2 = self.find_element(drop_on_id)
        x1, y1 = node1['location']
        x2, y2 = node2['location']
        command = "import pyautogui; "

        command += f"pyautogui.moveTo({x1}, {y1}); "
        # TODO: specified duration?
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1.); pyautogui.mouseUp(); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

        # Return pyautoguicode to drag and drop the elements

        return command

    @agent_action
    def scroll(self, element_id: int, clicks: int):
        """Scroll the element in the specified direction
        Args:
            element_id:int ID of the element to scroll in
            clicks:int the number of clicks to scroll can be positive (up) or negative (down).
        """
        try:
            node = self.find_element(element_id)
        except:
            node = self.find_element(0)
        # print(node.attrib)
        # coordinates = eval(
        #     node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)")
        # )
        # sizes = eval(node.get("{{{:}}}size".format(component_ns), "(-1, -1)"))

        # # Calculate the center of the element
        # x = coordinates[0] + sizes[0] // 2
        # y = coordinates[1] + sizes[1] // 2
        x, y = node['location']
        return (
            f"import pyautogui; pyautogui.moveTo({x}, {y}); pyautogui.scroll({clicks})"
        )

    @agent_action
    def hotkey(self, keys: List):
        """Press a hotkey combination
        Args:
            keys:List the keys to press in combination in a list format (e.g. ['ctrl', 'c'])
        """
        # add quotes around the keys
        keys = [f"'{key}'" for key in keys]
        return f"import pyautogui; pyautogui.hotkey({', '.join(keys)})"

    @agent_action
    def hold_and_press(self, hold_keys: List, press_keys: List):
        """Hold a list of keys and press a list of keys
        Args:
            hold_keys:List, list of keys to hold
            press_keys:List, list of keys to press in a sequence
        """

        press_keys_str = "[" + ", ".join([f"'{key}'" for key in press_keys]) + "]"
        command = "import pyautogui; "
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.press({press_keys_str}); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

        return command

    @agent_action
    def wait(self, time: float):
        """Wait for a specified amount of time
        Args:
            time:float the amount of time to wait in seconds
        """
        return f"""import time; time.sleep({time})"""

    @agent_action
    def done(self):
        """End the current task with a success"""
        return """DONE"""

    @agent_action
    def fail(self):
        """End the current task with a failure"""
        return """FAIL"""
