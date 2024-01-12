
def extract_frame_list_from_cfg(key_frames, select_clip, frame_num_per_clip, frame_num):
    l = [0] + key_frames
    if 'all' in select_clip:
        keyframe_list_index = list(range(len(key_frames)))
    elif 'longest' in select_clip:
        clip_length = [l[i+1] - l[i] for i in range(len(key_frames))]
        longest_index = clip_length.index(max(clip_length))
        keyframe_list_index = [longest_index]
    else:
        clip_num = int(select_clip[:-4])
        clip_num = max(int(frame_num*clip_num/(60*25)) ,1)

        if len(key_frames) < clip_num:
            keyframe_list_index = list(range(len(key_frames)))
        else:
            keyframe_list_index = []
            for clip_index in range(clip_num):
                cur_index = (len(key_frames)//clip_num) * clip_index
                keyframe_list_index.append(cur_index)
    if frame_num_per_clip == 'all':
        cal_frame_list = [list(range(l[i], l[i+1])) for i in keyframe_list_index]
    else:
        frame_num_per_clip += 2
        cal_frame_list = [list(range(l[i], l[i+1], max((l[i+1]-l[i]+1)//int(frame_num_per_clip),1)))[1 :-1] for i in keyframe_list_index]
    new_cal_frame_list = []
    new_keyframe_list_index = []
    for keyframe_index, frame_list in zip(keyframe_list_index, cal_frame_list):
        if len(frame_list) != 0:
            new_cal_frame_list.append(frame_list)
            new_keyframe_list_index.append(keyframe_index)
    return new_keyframe_list_index, new_cal_frame_list, l