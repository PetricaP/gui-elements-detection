import argparse
import json
import tkinter as tk
from tkinter import filedialog

import cv2

from detection import detect_rectangles, detect_text, join_padded_rectangles, detect_buttons, detect_check_buttons, \
    detect_radial_buttons, apply_ocr_on_rectangle
from utils import rectangle
from visualize_results import visualize_results


def analyze_image(image, model_path):
    net_results = detect_text(image, model_path, 0.1)

    text_rects = join_padded_rectangles(net_results, (0.05, 0.05), image.shape[:2])

    results = {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    rectangles = detect_rectangles(processed, 50, 0.0)

    buttons = detect_buttons(image, rectangles, text_rects)
    results['buttons'] = buttons

    check_buttons = detect_check_buttons(gray, rectangles, text_rects)
    results['check_buttons'] = check_buttons

    radial_buttons = detect_radial_buttons(gray, text_rects)
    results['radial_buttons'] = radial_buttons

    processed_text_rects = []
    for button in results['buttons']:
        text_rect = rectangle.from_json(button['rectangle'])

        processed_text_rects.append(text_rect)
        text = apply_ocr_on_rectangle(image, text_rect, (0.0, 0.0))

        button['text'] = text

    for check_button in results['check_buttons']:
        text_rect = check_button['associated_text_rect']

        if text_rect:
            text_rect = rectangle.from_json(text_rect)

            processed_text_rects.append(text_rect)
            text = apply_ocr_on_rectangle(image, text_rect, (0.1, 0.1))
        else:
            text = None

        check_button['text'] = text

    for radial_button in radial_buttons:
        text_rect = rectangle.from_json(radial_button['associated_text_rect'])

        processed_text_rects.append(text_rect)
        text = apply_ocr_on_rectangle(image, text_rect, (0.1, 0.1))

        radial_button['text'] = text

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
        self._widget_frame = tk.Frame(master=self, bg='#555555')
        self._widget_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        # The entry for the image path to be loaded and displayed, used as input to the analyzer
        self._image_path_entry_var = tk.StringVar()
        self._image_path_entry_var.set(image_path)

        self._image_path_label_frame = tk.LabelFrame(self._widget_frame, text='Image Path', bg='#666666', fg='#cccccc')
        self._image_path_label_frame.grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)

        self._image_path_entry = tk.Entry(self._image_path_label_frame, textvariable=self._image_path_entry_var,
                                          width=60)
        self._image_path_entry.grid(row=0, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5, pady=5)

        self._load_image_button = tk.Button(self._image_path_label_frame, text='Choose Image',
                                            command=self.choose_image,
                                            bg='#222222', fg='#ffffff')
        self._load_image_button.grid(row=1, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5, pady=5)

        # The entry for the path of the DNN model to use
        self._model_path_label_frame = tk.LabelFrame(self._widget_frame, text='DNN trained network path', bg='#666666',
                                                     fg='#cccccc')
        self._model_path_label_frame.grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)

        self._model_path_entry_var = tk.StringVar()
        self._model_path_entry_var.set(model_path)

        self._model_path_entry = tk.Entry(self._model_path_label_frame, textvariable=self._model_path_entry_var,
                                          width=60)
        self._model_path_entry.grid(row=0, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5, pady=5)

        self._model_image_button = tk.Button(self._model_path_label_frame, text='Choose PB file',
                                             command=self.choose_model, bg='#222222', fg='#ffffff')
        self._model_image_button.grid(row=1, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5, pady=5)

        # The entry for the path where the resulting json file will be saved
        self._results_path_label_frame = tk.LabelFrame(self._widget_frame, text='DNN trained network path',
                                                       bg='#666666',
                                                       fg='#cccccc')
        self._results_path_label_frame.grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)

        self._results_path_entry_var = tk.StringVar()
        self._results_path_entry_var.set(output_path)

        self._results_path_entry = tk.Entry(self._results_path_label_frame, textvariable=self._results_path_entry_var,
                                            width=60)
        self._results_path_entry.grid(row=0, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5, pady=5)

        # The button to RUN the gui analyzer on the image
        self._run_button = tk.Button(self._widget_frame, text='RUN',
                                     command=self.run, bg='#222222', fg='#ffffff')
        self._run_button.grid(row=4, column=0, sticky=tk.S, ipadx=20, ipady=10, padx=5, pady=20)

        # The button to visualize the results
        self._visualize_button = tk.Button(self._widget_frame, text='Visualize results',
                                           command=self.gui_visualize_results, bg='#222222', fg='#ffffff')
        self._visualize_button.grid(row=5, column=0, sticky=tk.S, ipadx=5, ipady=5, padx=5, pady=10)

        # The check box to specify whether the visualization image result needs to be saved
        self._visualize_image_save_check_var = tk.IntVar()
        self._visualize_image_save_check = tk.Checkbutton(self._widget_frame,
                                                          text='Save the visualization image',
                                                          variable=self._visualize_image_save_check_var,
                                                          command=self.on_check_button)
        self._visualize_image_save_check.grid(row=6, sticky=tk.S)

        # The path for the image in which the visualization will be saved
        self._visualize_image_save_path_label_frame = tk.LabelFrame(self._widget_frame,
                                                                    text='Path to the image in which to save the '
                                                                         'visualization results',
                                                                    bg='#666666',
                                                                    fg='#cccccc')
        self._visualize_image_save_path_label_frame.grid(row=7, column=0, sticky=tk.W, padx=10, pady=10)

        self._visualize_image_save_path_entry_var = tk.StringVar()
        self._visualize_image_save_path_entry_var.set("results.png")

        self._visualize_image_save_path_entry = tk.Entry(self._visualize_image_save_path_label_frame,
                                                         textvariable=self._visualize_image_save_path_entry_var,
                                                         width=60)
        self._visualize_image_save_path_entry.grid(row=0, column=0, sticky=tk.W, ipadx=5, ipady=5, padx=5, pady=5)
        self._visualize_image_save_path_entry['state'] = 'disabled'

    def choose_image(self):
        entry_image_path = self._image_path_entry_var.get()
        if not entry_image_path:
            image_path = filedialog.askopenfilename(initialdir='./test_images', title="Select input image")
            if image_path:
                try:
                    self._image = tk.PhotoImage(file=image_path)
                    self._input_image_canvas.create_image(5, 5, anchor=tk.NW, image=self._image)
                except Exception:
                    cv2.imshow("Input image", cv2.imread(image_path))
        else:
            image_path = entry_image_path
            try:
                self._image = tk.PhotoImage(file=image_path)
                self._input_image_canvas.create_image(5, 5, anchor=tk.NW, image=self._image)
            except Exception:
                try:
                    cv2.imshow("Input image", cv2.imread(image_path))
                except Exception:
                    # TODO: Show popup error message
                    return

        self._image_path_entry_var.set(entry_image_path)

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
        image = cv2.imread(self._image_path_entry_var.get())
        results = analyze_image(image, self._model_path_entry_var.get())

        with open(self._results_path_entry_var.get(), 'w') as file:
            file.write(json.dumps(results, indent=2))

    def gui_visualize_results(self):
        result_image = visualize_results(self._image_path_entry_var.get(), self._results_path_entry_var.get())

        cv2.imshow("Result", result_image)

        if self._visualize_image_save_check_var.get():
            cv2.imwrite(self._visualize_image_save_path_entry_var.get(), result_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to the pb file for the neural network')
    parser.add_argument('image_path', help='Path to the image to analyze')
    parser.add_argument('--output', '-o', help='Path to the results file to write to', default='results.json')
    parser.add_argument('--gui', action='store_true', help='Launch GUI for application')
    args = parser.parse_args()

    if not args.gui:
        image = cv2.imread(args.image_path)
        results = analyze_image(image, args.model_path)

        with open(args.output, 'w') as file:
            file.write(json.dumps(results, indent=2))
    else:
        main_window = tk.Tk()
        main_window.title('GUI Analyzer')
        main_window.geometry('1080x720')

        Application(main_window, args.image_path, args.output, args.model_path)

        main_window.mainloop()


if __name__ == '__main__':
    main()
