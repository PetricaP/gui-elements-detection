import json
import os
import sys
import tkinter as tk
from tkinter import filedialog
import logging as log

import cv2

from detection import detect_rectangles, detect_text, join_padded_rectangles, detect_buttons, detect_check_buttons, \
    detect_radial_buttons, apply_ocr_on_rectangle, apply_ocr_on_rects
from utils import rectangle
from visualize_results import visualize_results

FORMAT = '[%(asctime)s] [%(levelname)s] : %(message)s'
log.basicConfig(stream=sys.stdout, level=log.DEBUG, format=FORMAT)


def analyze_image(image, model_path, debug=False):
    net_results = detect_text(image, model_path, 0.1)

    text_rects = join_padded_rectangles(net_results, (0.05, 0.05), image.shape[:2])

    results = {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    rectangles = detect_rectangles(processed, 50, 0.0)
    if debug:
        ocr_results = apply_ocr_on_rects(gray, text_rects, (0.1, 0.1))
        debug_json = {
            'rectangles': [rect.to_json() for rect in rectangles],
            'texts': [{'text': result[1], 'rectangle': result[0].to_json()} for result in ocr_results]
        }
        with open('debug.json', 'w') as file:
            file.write(json.dumps(debug_json, indent=2))

    buttons = detect_buttons(image, rectangles, text_rects)
    results['buttons'] = buttons
    log.info("Completed detect buttons function")

    check_buttons = detect_check_buttons(gray, rectangles, text_rects)
    results['check_buttons'] = check_buttons
    log.info("Completed detect check buttons function")

    radial_buttons = detect_radial_buttons(gray, text_rects)
    results['radial_buttons'] = radial_buttons
    log.info("Completed detect radial buttons function")

    processed_text_rects = []
    for button in results['buttons']:
        text_rect = rectangle.from_json(button['rectangle'])

        processed_text_rects.append(text_rect)
        text = apply_ocr_on_rectangle(image, text_rect, (0.0, 0.0))

        button['text'] = text
    log.info("Applying ocr on buttons finished")

    for check_button in results['check_buttons']:
        text_rect = check_button['associated_text_rect']

        if text_rect:
            text_rect = rectangle.from_json(text_rect)

            processed_text_rects.append(text_rect)
            text = apply_ocr_on_rectangle(image, text_rect, (0.1, 0.1))
        else:
            text = None

        check_button['text'] = text
    log.info("Applying ocr on check buttons finished")

    for radial_button in radial_buttons:
        text_rect = rectangle.from_json(radial_button['associated_text_rect'])

        processed_text_rects.append(text_rect)
        text = apply_ocr_on_rectangle(image, text_rect, (0.1, 0.1))

        radial_button['text'] = text
    log.info("Applying ocr on radial buttons finished")

    return results


class Application(tk.Frame):
    def __init__(self, master, image_path, output_path, model_path='./other/frozen_east_text_detection.pb'):
        tk.Frame.__init__(self, master, bg='#777777')
        self.pack(fill=tk.BOTH, expand=1)

        self._input_image_canvas = tk.Canvas(self, width=640, height=640)
        self._input_image_canvas.pack(side=tk.LEFT)

        if image_path:
            self._image = tk.PhotoImage(file=image_path)
            self._input_image_canvas.create_image(5, 5, anchor=tk.NW, image=self._image)

        # This frame will hold all of the widgets on the right side of the gui
        widget_frame = tk.Frame(master=self, bg='#555555')
        widget_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        # The entry for the image path to be loaded and displayed, used as input to the analyzer
        self._image_path_entry_var = tk.StringVar()
        self._image_path_entry_var.set(image_path)

        image_path_label_frame = tk.LabelFrame(widget_frame, text='Image Path', bg='#666666', fg='#cccccc')
        image_path_label_frame.grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)

        tk.Entry(image_path_label_frame, textvariable=self._image_path_entry_var, width=60).grid(row=0, column=0,
                                                                                                 sticky=tk.W, ipadx=5,
                                                                                                 ipady=5, padx=5,
                                                                                                 pady=5)

        tk.Button(image_path_label_frame, text='Choose Image', command=self.choose_image, bg='#222222',
                  fg='#ffffff').grid(row=1, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5, pady=5)

        # The entry for the path of the DNN model to usef
        model_path_label_frame = tk.LabelFrame(widget_frame, text='DNN trained network path', bg='#666666',
                                               fg='#cccccc')
        model_path_label_frame.grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)

        self._model_path_entry_var = tk.StringVar()
        self._model_path_entry_var.set(model_path)

        tk.Entry(model_path_label_frame, textvariable=self._model_path_entry_var, width=60).grid(row=0, column=0,
                                                                                                 sticky=tk.W,
                                                                                                 ipadx=5, ipady=5,
                                                                                                 padx=5, pady=5)

        tk.Button(model_path_label_frame, text='Choose PB file',
                  command=self.choose_model, bg='#222222', fg='#ffffff').grid(row=1, column=0, sticky=tk.W, ipadx=5,
                                                                              ipady=5, padx=5, pady=5)

        # The entry for the path where the resulting json file will be saved
        results_path_label_frame = tk.LabelFrame(widget_frame,
                                                 text='Path to the output json file',
                                                 bg='#666666',
                                                 fg='#cccccc')
        results_path_label_frame.grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)

        self._results_path_entry_var = tk.StringVar()
        self._results_path_entry_var.set(output_path)

        tk.Entry(results_path_label_frame, textvariable=self._results_path_entry_var, width=60).grid(row=0, column=0,
                                                                                                     sticky=tk.W,
                                                                                                     ipadx=5, ipady=5,
                                                                                                     padx=5, pady=5)

        action_frame = tk.Frame(widget_frame, bg='#555555')
        action_frame.grid(row=4, column=0, sticky=tk.S)

        # The button to RUN the gui analyzer on the image
        tk.Button(action_frame, text='RUN', command=self.run, bg='#222222', fg='#ffffff').grid(row=0, column=0,
                                                                                               sticky=tk.W, ipadx=5,
                                                                                               ipady=5,
                                                                                               padx=50)

        # The button to visualize the results
        tk.Button(action_frame, text='Visualize results', command=self.gui_visualize_results, bg='#222222',
                  fg='#ffffff').grid(row=0, column=1, sticky=tk.W, ipadx=5, ipady=5, padx=100)

        self._check_button_frame = tk.Frame(widget_frame, bg='#555555')
        self._check_button_frame.grid(row=5, column=0, sticky=tk.W, padx=10, pady=20)

        # The check box to specify whether the visualization image result needs to be saved
        self._visualize_image_save_check_var = tk.IntVar()
        tk.Checkbutton(self._check_button_frame,
                       text='Save the visualization image',
                       variable=self._visualize_image_save_check_var,
                       command=self.on_check_button).grid(row=0, column=0, sticky=tk.W)

        # The path for the image in which the visualization will be saved
        visualize_image_save_path_label_frame = tk.LabelFrame(widget_frame,
                                                              text='Path to the image in which to save the '
                                                                   'visualization results',
                                                              bg='#666666',
                                                              fg='#cccccc')
        visualize_image_save_path_label_frame.grid(row=6, column=0, sticky=tk.W, padx=10, pady=10)

        self._visualize_image_save_path_entry_var = tk.StringVar()
        self._visualize_image_save_path_entry_var.set("results.png")

        self._visualize_image_save_path_entry = tk.Entry(visualize_image_save_path_label_frame,
                                                         textvariable=self._visualize_image_save_path_entry_var,
                                                         width=60)
        self._visualize_image_save_path_entry.grid(row=0, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5, pady=5)
        self._visualize_image_save_path_entry['state'] = 'disabled'

        # Debug check button
        self._debug_check_button_var = tk.IntVar()
        tk.Checkbutton(self._check_button_frame, text='Debug info', variable=self._debug_check_button_var) \
            .grid(row=1, column=0, sticky=tk.W, pady=10)

        Application._setup_legend(widget_frame)

        # progress var
        self.run_progress_var = tk.DoubleVar()
        self.run_progress_var.set(20)

    @staticmethod
    def _setup_legend(widget_frame):
        legend_frame = tk.Frame(widget_frame, bg='#555555')
        legend_frame.grid(row=7, sticky=tk.W, pady=10)

        # The legend for the visualization tool
        legend_label_frame = tk.LabelFrame(legend_frame, text='Visualization tool Legend')
        legend_label_frame.grid(row=0, column=0, sticky=tk.W, padx=10)

        tk.Frame(legend_label_frame, bg='#00ff00').grid(row=0, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5)
        tk.Label(legend_label_frame, text='-  Button').grid(row=0, column=1, sticky=tk.W)

        tk.Frame(legend_label_frame, bg='#0000ff').grid(row=1, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5)
        tk.Label(legend_label_frame, text='-  Checked Check Button').grid(row=1, column=1, sticky=tk.W)

        tk.Frame(legend_label_frame, bg='#6464ff').grid(row=2, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5)
        tk.Label(legend_label_frame, text='-  Unchecked Check Button').grid(row=2, column=1, sticky=tk.W)

        tk.Frame(legend_label_frame, bg='#ff6400').grid(row=3, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5)
        tk.Label(legend_label_frame, text='-  Checked Radial Button').grid(row=3, column=1, sticky=tk.W)

        tk.Frame(legend_label_frame, bg='#ffdc00').grid(row=4, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5)
        tk.Label(legend_label_frame, text='-  Unchecked Radial Button').grid(row=4, column=1, sticky=tk.W)

        # The legend for debug info
        debug_legend_label_frame = tk.LabelFrame(legend_frame, text='Debug info legend')
        debug_legend_label_frame.grid(row=0, column=1, sticky=tk.N, padx=20)

        tk.Frame(debug_legend_label_frame, bg='#690000').grid(row=0, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5)
        tk.Label(debug_legend_label_frame, text='-  Rectangle').grid(row=0, column=1, sticky=tk.W)

        tk.Frame(debug_legend_label_frame, bg='#006969').grid(row=1, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5)
        tk.Label(debug_legend_label_frame, text='-  Text rectangle').grid(row=1, column=1, sticky=tk.W)

    def choose_image(self):
        image_path = filedialog.askopenfilename(initialdir='./test_images', title="Select input image")
        if image_path:
            # noinspection PyBroadException
            try:
                self._image = tk.PhotoImage(file=image_path)
                self._input_image_canvas.create_image(5, 5, anchor=tk.NW, image=self._image)
            except Exception:
                cv2.imshow("Input image", cv2.imread(image_path))

        self._image_path_entry_var.set(image_path)

    def choose_model(self):
        entry_model_path = self._model_path_entry_var.get()
        if not entry_model_path:
            model_path = filedialog.askopenfilename(initialdir='./other', title="Select input image")
            if model_path:
                self._model_path_entry_var.set(model_path)

    def on_check_button(self):
        if self._visualize_image_save_check_var.get():
            self._visualize_image_save_path_entry['state'] = 'normal'
        else:
            self._visualize_image_save_path_entry['state'] = 'disabled'

    def run(self):
        from tkinter import messagebox
        try:
            image = cv2.imread(self._image_path_entry_var.get())
            height, width, _ = image.shape
            if height > 1000 or width > 1000:
                log.info("Resizing image (width or height > 1000)")
                scale_percent = 500 / max(height, width)
                print(scale_percent)
                width = int(width * scale_percent)
                height = int(height * scale_percent)
                dim = (width, height)
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            results = analyze_image(image, self._model_path_entry_var.get(), self._debug_check_button_var.get())

            with open(self._results_path_entry_var.get(), 'w') as file:
                file.write(json.dumps(results, indent=2))
        except cv2.error:
            messagebox.showinfo("Run error", "Could not find pb model")
        except AttributeError:
            messagebox.showinfo("Run error", "Could not find image path")

    def gui_visualize_results(self):
        result_image = visualize_results(self._image_path_entry_var.get(), self._results_path_entry_var.get(),
                                         self._debug_check_button_var.get())

        cv2.imshow("Result", result_image)

        if self._visualize_image_save_check_var.get():
            output_image = self._visualize_image_save_path_entry_var.get()
            if self._debug_check_button_var.get():
                base, ext = os.path.splitext(output_image)
                output_image = base + '_debug' + ext

            cv2.imwrite(output_image, result_image)
