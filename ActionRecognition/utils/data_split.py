import pandas as pd
from sklearn.model_selection import train_test_split

def split_data():
    # CSV 파일을 읽어들입니다.
    input_csv = './output/merged_output.csv'  # 여기서 'input.csv'를 실제 입력 파일명으로 대체하세요
    df = pd.read_csv(input_csv)

    # 데이터를 90%는 train, 10%는 test로 분할합니다.
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # train.csv와 test.csv로 저장합니다.
    train_df.to_csv('./output/train.csv', index=False)
    test_df.to_csv('./output/test.csv', index=False)

    print("train.csv와 test.csv 파일이 성공적으로 저장되었습니다.")

def convert_txt():
    # CSV 파일을 읽어들입니다.
    input_csv = './output/train.csv'  # 여기를 실제 파일 이름으로 변경하세요
    df = pd.read_csv(input_csv)

    # TXT 파일로 변환하여 저장합니다.
    output_txt = './output/train.txt'  # 원하는 출력 파일 이름으로 변경하세요
    with open(output_txt, 'w') as file:
        for index, row in df.iterrows():
            file.write(f"{row['video_path']} {row['action_index']}\n")

    print("TXT 파일로 성공적으로 변환되었습니다.")


# 파일을 읽고 새로운 내용을 저장하는 함수입니다.
def process_file():
    # 파일 경로를 지정합니다.
    input_file = './output/test.txt'
    output_file = './output/test_2.txt'
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parts = line.strip().split(' ')
            # 경로를 분리하고 output 이후의 내용을 추출합니다.
            path = parts[0].split('/output/')[-1]
            # 숫자를 붙여서 새로운 내용을 만듭니다.
            new_line = f'{path} {parts[1]}\n'
            outfile.write(new_line)

# 함수 호출
process_file()
