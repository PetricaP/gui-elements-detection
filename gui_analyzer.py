import argparse
import json
import tkinter as tk

import cv2

from application import Application, analyze_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to the pb file for the neural network')
    parser.add_argument('image_path', help='Path to the image to analyze')
    parser.add_argument('--output', '-o', help='Path to the results file to write to', default='results.json')
    parser.add_argument('--gui', action='store_true', help='Launch GUI for application')
    parser.add_argument('--debug', action='store_true', help='Store intermediate results in debug.json')
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
