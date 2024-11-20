import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("desktopenv.agent")


# Agent action decorator
def agent_action(func):
    func.is_agent_action = True
    return func


class GroundingAgent:
    def __init__(self):
        
        self.index_out_of_range_flag = False
        self.x_scale = 1920 / 1366
        self.y_scale = 1080 / 768
        # self.top_app = None
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
        # global state_ns, component_ns, attributes_ns, value_ns
        # state_ns = "uri:deskat:state.at-spi.gnome.org"

        self.notes = []
        self.clipboard = ""

        # TODO: this is terrible, fix this
    def get_current_applications(self, obs):
        tree = ET.ElementTree(ET.fromstring(obs["accessibility_tree"]))
        apps = []
        exclude_list = ["gjs", "gnome-shell"]
        for node in tree.iter():
            # Keep applications and only those which have children
            if (
                node.tag.endswith("application")
                and list(node)
                and node.attrib.get("name", "") not in exclude_list
            ):
                apps.append(node.attrib.get("name", "").replace("\\", ""))
        return apps
    def resize_coordinates(self, coordinates: List):
        x, y = coordinates

        x_new = int(x * self.x_scale)
        y_new = int(y * self.y_scale)

        logger.info(f"The original coordinates is {coordinates}, the scaled one is {[x_new, y_new]}")

        return[x_new, y_new]
     
    @agent_action
    def click(
        self,
        element_coordinates: List,
        num_clicks: int = 1,
        button_type: str = "left",
        hold_keys: List = [],
    ):
        """Click on the element
        Args:
            element_coordinates:List: [x, y], coordinates of the element to click on
            num_clicks:int, number of times to click the element
            button_type:str, which mouse button to press can be "left", "middle", or "right"
            hold_keys:List, list of keys to hold while clicking
        """
        x, y = self.resize_coordinates(element_coordinates)
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
        element_coordinates: Optional[List] = None,
        text: str = '',
        overwrite: bool = False,
        enter: bool = False,
    ):
        """Type text into the element
        Args:
            element_coordinates:**List: [x, y]**, coordiates of the element to type into. If not provided, typing will start at the current cursor location.
            text:str, the text to type
            overwrite:bool, Assign it to True if the text should overwrite the existing text, otherwise assign it to False. Using this argument clears all text in an element.
            enter:bool, Assign it to True if the enter key should be pressed after typing the text, otherwise assign it to False.
        """
        
        if element_coordinates is not None:
            # If a node is found, retrieve its coordinates and size
            # Start typing at the center of the element

            x, y = self.resize_coordinates(element_coordinates)

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
    def drag_and_drop(self, drag_from_coordinates: List, drop_on_coordinates: List, hold_keys: List = []):
        """Drag element1 and drop it on element2.
        Args:
            drag_from_coordinates:List: [x, y], coordinates of element to drag
            drop_on_coordinates:List: [x, y], coordinates of element to drop on
            hold_keys:List list of keys to hold while dragging
        """
        x1, y1 = self.resize_coordinates(drag_from_coordinates)
        x2, y2 = self.resize_coordinates(drop_on_coordinates)

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
    def scroll(self, element_coordinates: List, clicks: int):
        """Scroll the element in the specified direction
        Args:
            element_coordinates:List: [x, y], coordinates of the element to scroll in
            clicks:int, the number of clicks to scroll can be positive (up) or negative (down).
        """

        x, y = self.resize_coordinates(element_coordinates)

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
