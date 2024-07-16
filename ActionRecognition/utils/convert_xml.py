import cv2
import xml.etree.ElementTree as ET
import csv
import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processor")

        self.input_folder = ""
        self.output_folder = ""
        self.csv_files = []

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Select Input Folder:").grid(row=0, column=0, padx=10, pady=10)
        self.input_button = tk.Button(self.root, text="Browse", command=self.select_input_folder)
        self.input_button.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(self.root, text="Select Output Folder:").grid(row=1, column=0, padx=10, pady=10)
        self.output_button = tk.Button(self.root, text="Browse", command=self.select_output_folder)
        self.output_button.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(self.root, text="Select CSV Files to Merge:").grid(row=2, column=0, padx=10, pady=10)
        self.csv_button = tk.Button(self.root, text="Browse", command=self.select_csv_files)
        self.csv_button.grid(row=2, column=1, padx=10, pady=10)

        self.run_button = tk.Button(self.root, text="Run", command=self.run_processing)
        self.run_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        self.merge_button = tk.Button(self.root, text="Merge CSVs", command=self.merge_csv_files)
        self.merge_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    def select_input_folder(self):
        self.input_folder = filedialog.askdirectory()
        if self.input_folder:
            print(f"Selected input folder: {self.input_folder}")

    def select_output_folder(self):
        self.output_folder = filedialog.askdirectory()
        if self.output_folder:
            print(f"Selected output folder: {self.output_folder}")

    def select_csv_files(self):
        self.csv_files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if self.csv_files:
            print(f"Selected CSV files: {self.csv_files}")

    def run_processing(self):
        if not self.input_folder or not self.output_folder:
            messagebox.showerror("Error", "Please select both input and output folders")
            return
        
        self.main(self.input_folder, self.output_folder)
        messagebox.showinfo("Success", "Processing completed!")

    def merge_csv_files(self):
        if not self.csv_files or not self.output_folder:
            messagebox.showerror("Error", "Please select CSV files and an output folder")
            return
        
        merged_data = []
        for csv_file in self.csv_files:
            with open(csv_file, 'r', newline='') as file:
                reader = csv.reader(file)
                headers = next(reader)
                if not merged_data:
                    merged_data.append(headers)
                merged_data.extend(row for row in reader)
        
        output_file = os.path.join(self.output_folder, 'merged_output.csv')
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(merged_data)

        print(f"Merged CSV files saved to: {output_file}")
        messagebox.showinfo("Success", f"CSV 파일이 병합되었습니다.: {output_file}")

    def parse_annotation(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        video_info = {
            "filename": os.path.join(self.input_folder, root.find("filename").text),
            "actions": []
        }
        
        action_set = set()
        
        for action in root.findall(".//action"):
            action_name = action.find("actionname").text
            action_set.add(action_name)
            action_frames = [(int(frame.find("start").text), int(frame.find("end").text)) for frame in action.findall("frame")]
            video_info["actions"].append({"name": action_name, "frames": action_frames})
        
        return video_info, action_set

    def save_video_segment(self, cap, start_frame, end_frame, output_path):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out = None
        frame_idx = start_frame
        
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
            
            out.write(frame)
            frame_idx += 1
        
        if out:
            out.release()

    def process_video(self, xml_file, action_index):
        video_info, _ = self.parse_annotation(xml_file)
        video_path = video_info["filename"]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        os.makedirs(self.output_folder, exist_ok=True)
        csv_data = []
        
        for action in video_info["actions"]:
            action_name = action["name"]
            action_idx = action_index[action_name]
            
            for start_frame, end_frame in action["frames"]:
                output_filename = f"{self.output_folder}/{os.path.splitext(os.path.basename(video_path))[0]}_{start_frame}_{end_frame}.mp4"
                self.save_video_segment(cap, start_frame, end_frame, output_filename)
                csv_data.append([output_filename, action_idx])
        
        cap.release()
        
        print(f"Finished processing video: {video_path}")
        return csv_data

    def main(self, input_folder, output_folder):
        xml_files = glob.glob(os.path.join(input_folder, "*.xml"))
        all_csv_data = []
        all_action_set = set()
        
        # Collect all actions from XML files
        for xml_file in xml_files:
            _, action_set = self.parse_annotation(xml_file)
            all_action_set.update(action_set)
        
        # Create action index dictionary
        action_dict = {idx: action for idx, action in enumerate(sorted(all_action_set))}
        action_index = {action: idx for idx, action in action_dict.items()}
        
        # Process each video
        for xml_file in xml_files:
            csv_data = self.process_video(xml_file, action_index)
            if csv_data:
                all_csv_data.extend(csv_data)
        
        # Save action dictionary to a file
        with open(os.path.join(output_folder, 'action_dict.csv'), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['action_index', 'action_name'])
            for idx, action in action_dict.items():
                csvwriter.writerow([idx, action])
        
        # Save video actions to a file
        with open(os.path.join(output_folder, 'actions.csv'), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['video_path', 'action_index'])
            csvwriter.writerows(all_csv_data)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()
