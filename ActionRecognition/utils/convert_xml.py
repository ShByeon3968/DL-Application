import xml.etree.ElementTree as ET

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    video_info = {
        "filename": root.find("filename").text,
    }

    actions = []

    for obj in root.findall("object"):
        person_name = obj.find("objectname").text

        for action in obj.findall("action"):
            action_name = action.find("actionname").text
            for frame in action.findall("frame"):
                start_frame = int(frame.find("start").text)
                end_frame = int(frame.find("end").text)
                actions.append((video_info["filename"], start_frame, end_frame, action_name))
    
    return actions

def save_to_file(actions, output_file):
    with open(output_file, 'w') as f:
        for action in actions:
            line = f"{action[0]} {action[1]} {action[2]} {action[3]}\n"
            f.write(line)

if __name__ == "__main__":
    xml_file = "input.xml"  # XML 파일 경로
    output_file = "output.txt"  # 출력 파일 경로

    actions = parse_annotation(xml_file)
    save_to_file(actions, output_file)

    print(f"Annotation data has been saved to {output_file}")
