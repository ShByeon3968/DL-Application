import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import cv2
import threading
from PIL import Image, ImageTk
import glob
import os
import numpy as np
import os
from depth2mesh import depth_to_mesh
import open3d as o3d

class DepthAnythingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Estimaitor")
        self.video_playing = False
        self.video_paused = False
        self.video_thread = None

        # 기본값 설정
        self.default_script_path = "C:/bsh/Python/DL-Application/Depth2Mesh/run_video.py"
        self.default_video_path = "path/to/your/video.mp4"
        self.default_outdir = "C:/bsh/Python/DL-Application/Depth2Mesh/output"
        self.default_encoder = "vits"
        self.default_input_size = 512
        

        # Configure grid layout
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Script path
        self.script_path_label = tk.Label(root, text="Python Script Path")
        self.script_path_label.grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.script_path_entry = tk.Entry(root, width=50)
        self.script_path_entry.grid(row=0, column=1, padx=10, pady=10, sticky='we')
        self.script_path_entry.insert(0, self.default_script_path)  # 기본값 설정
        self.browse_script_button = tk.Button(root, text="Browse", command=self.browse_script)
        self.browse_script_button.grid(row=0, column=2, padx=10, pady=10, sticky='w')

        # Video path
        self.video_path_label = tk.Label(root, text="Video Path")
        self.video_path_label.grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.video_path_entry = tk.Entry(root, width=50)
        self.video_path_entry.grid(row=1, column=1, padx=10, pady=10, sticky='we')
        self.video_path_entry.insert(0, self.default_video_path)  # 기본값 설정
        self.browse_video_button = tk.Button(root, text="Browse", command=self.browse_video)
        self.browse_video_button.grid(row=1, column=2, padx=10, pady=10, sticky='w')

        # Input size
        self.input_size_label = tk.Label(root, text="Input Size")
        self.input_size_label.grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.input_size_entry = tk.Entry(root, width=50)
        self.input_size_entry.grid(row=2, column=1, padx=10, pady=10, sticky='we')
        self.input_size_entry.insert(0, str(self.default_input_size))  # 기본값 설정

        # Output directory
        self.outdir_label = tk.Label(root, text="Output Directory")
        self.outdir_label.grid(row=3, column=0, padx=10, pady=10, sticky='e')
        self.outdir_entry = tk.Entry(root, width=50)
        self.outdir_entry.grid(row=3, column=1, padx=10, pady=10, sticky='we')
        self.outdir_entry.insert(0, self.default_outdir)  # 기본값 설정
        self.browse_outdir_button = tk.Button(root, text="Browse", command=self.browse_outdir)
        self.browse_outdir_button.grid(row=3, column=2, padx=10, pady=10, sticky='w')

        # Encoder, Pred only, Grayscale
        self.encoder_label = tk.Label(root, text="Encoder")
        self.encoder_label.grid(row=4, column=0, padx=10, pady=10, sticky='e')
        self.encoder_var = tk.StringVar(value=self.default_encoder)  # 기본값 설정
        self.encoder_options = ["vits", "vitb", "vitl", "vitg"]
        self.encoder_menu = tk.OptionMenu(root, self.encoder_var, *self.encoder_options)
        self.encoder_menu.grid(row=4, column=1, padx=10, pady=10, sticky='we')

        self.pred_only_var = tk.BooleanVar()
        self.pred_only_check = tk.Checkbutton(root, text="Prediction Only", variable=self.pred_only_var)
        self.pred_only_check.grid(row=4, column=2, padx=10, pady=10, sticky='w')

        self.grayscale_var = tk.BooleanVar()
        self.grayscale_check = tk.Checkbutton(root, text="Grayscale", variable=self.grayscale_var)
        self.grayscale_check.grid(row=4, column=3, padx=10, pady=10, sticky='w')

        # Run button
        self.run_button = tk.Button(root, text="Run", command=self.run_script)
        self.run_button.grid(row=5, column=0, columnspan=4, pady=20, sticky='we')

        # Video display area
        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.grid(row=6, column=1, columnspan=2, padx=10, pady=10, sticky='nsew')

        # Plot Video button
        self.plot_video_button = tk.Button(root, text="Plot Video", command=self.start_video_thread)
        self.plot_video_button.grid(row=7, column=0, padx=10, pady=10, sticky='w')

        # Play/Pause buttons
        self.play_button = tk.Button(root, text="Play", command=self.play_video, state='disabled')
        self.play_button.grid(row=7, column=2, padx=10, pady=10, sticky='e')

        self.pause_button = tk.Button(root, text="Pause", command=self.pause_video, state='disabled')
        self.pause_button.grid(row=7, column=3, padx=10, pady=10, sticky='w')

        # New Window button
        self.new_window_button = tk.Button(root, text="Start Mesh Generator", command=self.open_new_window)
        self.new_window_button.grid(row=8, column=0, columnspan=4, pady=20, sticky='we')

        # Exit button
        self.exit_button = tk.Button(root, text="Exit", command=self.quit_program)
        self.exit_button.grid(row=9, column=0, columnspan=4, pady=20, sticky='we')

        for i in range(10):
            self.root.grid_rowconfigure(i, weight=1)
        for i in range(4):
            self.root.grid_columnconfigure(i, weight=1)

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
        outdir = self.outdir_entry.get()
        if not outdir:
            messagebox.showerror("Error", "Please provide an output directory.")
            return

        video_file = self.get_latest_video_file(outdir)
        if not video_file:
            messagebox.showerror("Error", "No video file found in the output directory.")
            return

        self.video_playing = True
        self.video_thread = threading.Thread(target=self.plot_video, args=(video_file,))
        self.video_thread.start()
        self.play_button.config(state='normal')
        self.pause_button.config(state='normal')

    def get_latest_video_file(self, directory):
        video_files = glob.glob(os.path.join(directory, '*.mp4'))
        if not video_files:
            return None
        latest_video_file = max(video_files, key=os.path.getctime)
        return latest_video_file

    def plot_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open the video file.")
            return

        # 비디오 해상도를 가져와 Canvas 크기 설정
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.config(width=width//2, height=height//2)

        while cap.isOpened():
            if self.video_playing and not self.video_paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (width//2, height//2))
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                self.canvas.image = frame

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cap.release()

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

    def quit_program(self):
        self.video_playing = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join()
        self.root.quit()
        self.root.destroy()
        cv2.destroyAllWindows()

    def open_new_window(self):
        new_window = tk.Toplevel(self.root)
        original_x = self.root.winfo_x()
        original_y = self.root.winfo_y()
        new_window.geometry(f"{self.root.winfo_width()}x{self.root.winfo_height()}+{original_x + 500}+{original_y}")  # 원래 창과 동일한 크기, x 방향으로 100 이동
        MeshGeneratorGUI(new_window) 


class MeshGeneratorGUI:
    def __init__(self,root):
        self.root = root
        self.root.title("Mesh Generator")

        # 기본값 설정
        self.default_depthmap_path = "path/to/your/video.mp4"
        self.depthmap_list = []
        self.pcd_list = []
        self.res_pcd = None
        self.output_mesh = None

        # Configure grid layout
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Video path
        self.video_path_label = tk.Label(root, text="Video Path")
        self.video_path_label.grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.video_path_entry = tk.Entry(root, width=50)
        self.video_path_entry.grid(row=0, column=1, padx=10, pady=10, sticky='we')
        self.video_path_entry.insert(0, self.default_depthmap_path)  # 기본값 설정
        self.browse_video_button = tk.Button(root, text="Browse", command=self.browse_video)
        self.browse_video_button.grid(row=0, column=2, padx=10, pady=10, sticky='w')


        # Read Depth map button
        self.read_depth_button = tk.Button(root, text="Read Depth Map", command=self.read_depth_map)
        self.read_depth_button.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        # Generate Mesh button
        self.generate_mesh_button = tk.Button(root, text="Generate Mesh", command=self.depth2mesh)
        self.generate_mesh_button.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        # Show Mesh button
        self.show_mesh_button = tk.Button(root, text="Show Mesh", command=self.showMesh)
        self.show_mesh_button.grid(row=1, column=2, padx=10, pady=10, sticky='w')



    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4"), ("Image Files", "*.jpg;*.png")])
        if file_path:
            self.video_path_entry.delete(0, tk.END)
            self.video_path_entry.insert(0, file_path)

    def read_depth_map(self):
        video_path = self.video_path_entry.get()
        print("Processing...")
        if not video_path:
            messagebox.showerror("Error", "Please provide a video path.")
            return

        if video_path.lower().endswith(('.mp4')):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Failed to open the video file.")
                return

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.depthmap_list.append(frame)

            cap.release()
            messagebox.showinfo("Success", f"Read {len(self.depthmap_list)} frames from the video.")
        
        elif video_path.lower().endswith(('.jpg', '.png')):
            frame = cv2.imread(video_path)
            if frame is None:
                messagebox.showerror("Error", "Failed to open the image file.")
                return
            
            self.depthmap_list.append(frame)
            messagebox.showinfo("Success", f"Read 1 frame from the image file.")

        
        

    def depth2mesh(self):
        for depth in self.depthmap_list:
            depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
            depth = np.array(depth,dtype=np.float32)
            self.output_mesh = depth_to_mesh(depth=depth)
        messagebox.showinfo("Success", f"Mesh Generation is Done.")

    def showMesh(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.output_mesh.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.output_mesh.triangles)

        # Save the mesh to a file
        o3d.io.write_triangle_mesh("output.obj", mesh)
        messagebox.showinfo("Success", f"메쉬가 저장됨.")

        


if __name__ == "__main__":
    root = tk.Tk()
    gui = DepthAnythingGUI(root)
    root.mainloop()
