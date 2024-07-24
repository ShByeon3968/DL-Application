def read_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None

def txt_to_list(content):
    return content.splitlines()

if __name__ == "__main__":
    txt_file_path = "mmaction2/tools/data/kinetics/label_map_k400.txt"  # 읽어올 텍스트 파일 경로
    content = read_txt_file(txt_file_path)
    
    if content is not None:
        action_list = txt_to_list(content)
        print("List of actions:")
        print(action_list[288])
