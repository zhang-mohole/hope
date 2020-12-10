import numpy as np

def make_dir_path(root_path, with_object=False):
    pathes = []
    if with_object:
        with_object = root_path + "/data/withObject" # 188,050 frames in total
        for i in range(1, 184):
            end = 1025
            if i == 92:
                end = 477
            for j in range(1, end):

                pathes.append(with_object + "/{0:04d}/{1:04d}".format(i, j)) # 143,449 frames in total
    else:
        no_object = root_path + "/data/noObject"
        for i in range(1,141):
            end = 1025
            if i == 69:
                end = 217
            for j in range(1,end):
                pathes.append(no_object +"/{0:04d}/{1:04d}".format(i,j))

    return pathes

def get_pose_li(dir_path):
    joint_3d_global = []
    image_path = []

    for path in dir_path:
        image = path+"_color_composed.png"
        image_path.append(image)

        # value = open(path+"_joint_pos.txt").readline().strip('\n').split(',')
        # value = [float(val) for val in value]
        # joint_3d.append(value)

        value = open(path+"_joint_pos_global.txt").readline().strip('\n').split(',')
        value = [float(val) for val in value]
        joint_3d_global.append(value)

        # value = open(path+"_joint2D.txt").readline().strip('\n').split(',')
        # value = [float(val) for val in value]
        # joint_2d.append(value)

    return image_path, joint_3d_global

if __name__ == "__main__":
    with_object = True
    dir_path = make_dir_path('/home/tlh-lxy/zmh/data/GANerated_hand/GANerated/GANeratedHands_Release', with_object=with_object)
    image_path, joint_3d_global = get_pose_li(dir_path)

    joint_3d_global = np.array(joint_3d_global).reshape(-1, 21, 3)
    M0 = joint_3d_global[:, 9, :]
    W = joint_3d_global[:, 0, :]
    real_len = np.linalg.norm(W - M0, axis=1)
    scale = 1.0 / real_len
    print(scale[:3])

    scale_dict = zip(image_path, scale.tolist())
    scale_dict = dict(scale_dict)

    # 保存结果
    f = open('GANeratedHands_scale_withObject.txt','w')
    f.write(str(scale_dict))
    f.close()

    
    #读取
    # f = open('GANeratedHands_scale.txt','r')
    # a = f.read()
    # scale_dict = eval(a)
    # f.close()

    print('GANerated scale calculate done')