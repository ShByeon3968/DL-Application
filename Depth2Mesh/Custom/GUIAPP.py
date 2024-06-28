import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import cv2
import threading

class DepthAnythingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Anything V2 GUI")
        self.video_playing = False
        self.video_paused = False

        # Script path
        self.script_path_label = tk.Label(root, text="Python Script Path")
        self.script_path_label.grid(row=0, column=0, padx=10, pady=10)
        self.script_path_entry = tk.Entry(root, width=50)
        self.script_path_entry.grid(row=0, column=1, padx=10, pady=10)
        self.browse_script_button = tk.Button(root, text="Browse", command=self.browse_script)
        self.browse_script_button.grid(row=0, column=2, padx=10, pady=10)

        # Video path
        self.video_path_label = tk.Label(root, text="Video Path")
        self.video_path_label.grid(row=1, column=0, padx=10, pady=10)
        self.video_path_entry = tk.Entry(root, width=50)
        self.video_path_entry.grid(row=1, column=1, padx=10, pady=10)
        self.browse_video_button = tk.Button(root, text="Browse", command=self.browse_video)
        self.browse_video_button.grid(row=1, column=2, padx=10, pady=10)
        self.plot_video_button = tk.Button(root, text="Plot Video", command=self.start_video_thread)
        self.plot_video_button.grid(row=1, column=3, padx=10, pady=10)

        # Input size
        self.input_size_label = tk.Label(root, text="Input Size")
        self.input_size_label.grid(row=2, column=0, padx=10, pady=10)
        self.input_size_entry = tk.Entry(root, width=50)
        self.input_size_entry.grid(row=2, column=1, padx=10, pady=10)
        self.input_size_entry.insert(0, "518")

        # Output directory
        self.outdir_label = tk.Label(root, text="Output Directory")
        self.outdir_label.grid(row=3, column=0, padx=10, pady=10)
        self.outdir_entry = tk.Entry(root, width=50)
        self.outdir_entry.grid(row=3, column=1, padx=10, pady=10)
        self.browse_outdir_button = tk.Button(root, text="Browse", command=self.browse_outdir)
        self.browse_outdir_button.grid(row=3, column=2, padx=10, pady=10)

        # Encoder, Pred only, Grayscale
        self.encoder_label = tk.Label(root, text="Encoder")
        self.encoder_label.grid(row=4, column=0, padx=10, pady=10)
        self.encoder_var = tk.StringVar(value="vitl")
        self.encoder_options = ["vits", "vitb", "vitl", "vitg"]
        self.encoder_menu = tk.OptionMenu(root, self.encoder_var, *self.encoder_options)
        self.encoder_menu.grid(row=4, column=1, padx=10, pady=10, sticky='w')

        self.pred_only_var = tk.BooleanVar()
        self.pred_only_check = tk.Checkbutton(root, text="Prediction Only", variable=self.pred_only_var)
        self.pred_only_check.grid(row=4, column=1, padx=10, pady=10)

        self.grayscale_var = tk.BooleanVar()
        self.grayscale_check = tk.Checkbutton(root, text="Grayscale", variable=self.grayscale_var)
        self.grayscale_check.grid(row=4, column=2, padx=10, pady=10, sticky='w')

        # Run button
        self.run_button = tk.Button(root, text="Run", command=self.run_script)
        self.run_button.grid(row=5, column=0, columnspan=3, pady=20)

        # Play/Pause buttons
        self.play_button = tk.Button(root, text="Play", command=self.play_video, state='disabled')
        self.play_button.grid(row=6, column=1, padx=10, pady=10)

        self.pause_button = tk.Button(root, text="Pause", command=self.pause_video, state='disabled')
        self.pause_button.grid(row=6, column=2, padx=10, pady=10)

    def browse_script(self):
        file_path = filedialog.askopenfilename(filetypes=[("Main Script", "*.py")])
        if file_path:
            self.script_path_entry.delete(0, tk.END)
            self.script_path_entry.insert(0, file_path)

    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.*")])
        if file_path:
            self.video_path_entry.delete(0, tk.END)
            self.video_path_entry.insert(0, file_path)

    def browse_outdir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.outdir_entry.delete(0, tk.END)
            self.outdir_entry.insert(0, directory)

    def start_video_thread(self):
        video_path = self.video_path_entry.get()
        if not video_path:
            messagebox.showerror("Error", "Please provide a video path.")
            return

        self.video_playing = True
        self.video_thread = threading.Thread(target=self.plot_video, args=(video_path,))
        self.video_thread.start()
        self.play_button.config(state='normal')
        self.pause_button.config(state='normal')

    def plot_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open the video file.")
            return

        while cap.isOpened():
            if self.video_playing and not self.video_paused:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow('Video', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def play_video(self):
        self.video_paused = False

    def pause_video(self):
        self.video_paused = True

    def run_script(self):
        script_path = self.script_path_entry.get()
        video_path = self.video_path_entry.get()
        input_size = self.input_size_entry.get()
        outdir = self.outdir_entry.get()
        encoder = self.encoder_var.get()
        pred_only = self.pred_only_var.get()
        grayscale = self.grayscale_var.get()

        if not script_path or not video_path or not outdir:
            messagebox.showerror("Error", "Please provide all necessary inputs.")
            return

        command = [
            "python", script_path,
            "--video-path", video_path,
            "--input-size", input_size,
            "--outdir", outdir,
            "--encoder", encoder
        ]

        if pred_only:
            command.append("--pred-only")

        if grayscale:
            command.append("--grayscale")

        try:
            subprocess.run(command, check=True)
            messagebox.showinfo("Success", "Script executed successfully.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Script execution failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = DepthAnythingGUI(root)
    root.mainloop()
