import cv2
import xml.etree.ElementTree as ET
import csv
import os

# 0. 행동 이름 딕셔너리 만들기
action_dict = {
    0: 'pulling',
    1: 'threaten'
}

# 1. .mp4 파일의 경로를 받아 비디오 오픈
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    video_info = {
        "filename": os.path.join('./data/10.데이트폭력및추행(datefight)/inside_croki_04/593-4',root.find("filename").text),
        "actions": []
    }
    
    for action in root.findall(".//action"):
        action_name = action.find("actionname").text
        action_frames = [(int(frame.find("start").text), int(frame.find("end").text)) for frame in action.findall("frame")]
        video_info["actions"].append({"name": action_name, "frames": action_frames})
    
    return video_info

def save_video_segment(cap, start_frame, end_frame, output_path):
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

def main(xml_file):
    video_info = parse_annotation(xml_file)
    video_path = video_info["filename"]
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    os.makedirs("output", exist_ok=True)
    csv_data = []
    
    action_index = {v: k for k, v in action_dict.items()}
    
    for action in video_info["actions"]:
        action_name = action["name"]
        action_idx = action_index[action_name]
        
        for start_frame, end_frame in action["frames"]:
            output_filename = f"output/{os.path.splitext(os.path.basename(video_path))[0]}_{start_frame}_{end_frame}.mp4"
            save_video_segment(cap, start_frame, end_frame, output_filename)
            csv_data.append([output_filename, action_idx])
    
    cap.release()
    
    with open('output/actions.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['video_path', 'action_index'])
        csvwriter.writerows(csv_data)

if __name__ == "__main__":
    xml_file = "./data/10.데이트폭력및추행(datefight)/inside_croki_04/593-4/593-4_cam01_datefight02_place02_night_spring.xml"  # XML 파일 경로
    main(xml_file)
