from mmaction.apis import inference_recognizer, init_recognizer

action_dict = {
    0 : 'falldown',
    1 : 'kicking',
    2: 'pulling',
    3 : 'punching',
    4:'pushing',
    5 : 'threaten',
    6:'throwing'
}

config_path = 'mmaction2/configs/recognition/abuse/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_abuse-rgb.py'
checkpoint_path = './work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_abuse-rgb/best_acc_top1_epoch_46.pth' # can be a local path
img_path = './data/test_3/1-1_cam01_fight04_place02_night_summer_3546_3581.mp4'   # you can specify your own picture path

# build the model from a config file and a checkpoint file
model = init_recognizer(config_path, checkpoint_path, device="cuda:0")  # device can be 'cuda:0'
# test a single image
result = inference_recognizer(model, img_path)
action_num = result.pred_label.detach().cpu().numpy()[0]
print(action_dict[action_num])
