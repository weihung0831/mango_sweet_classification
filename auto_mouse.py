import pretty_errors
import rpa as r
import pyautogui
import time

# time.sleep(3)
#
# xloc, yloc = pyautogui.position()
# print(xloc, yloc)

for i in range(18):
    pyautogui.moveTo(418, 88, duration=0.00001)
    pyautogui.click()
    pyautogui.moveTo(418, 587, duration=0.00001)
    pyautogui.moveTo(1125, 587, duration=0.00001)
    pyautogui.click()
    pyautogui.moveTo(639, 963, duration=0.00001)
    pyautogui.click()
    pyautogui.moveTo(127, 911, duration=0.00001)
    pyautogui.click()
