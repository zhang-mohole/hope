import k_means

def make_dir_path(root_path):
    pathes = []
    no_object = root_path + "/data/noObject"
    for i in range(1,141):
        end = 1025
        if i == 69:
            end = 217
        for j in range(1,end):
            pathes.append(no_object +"/{0:04d}/{1:04d}".format(i,j))
    # with_object = root_path + "/data/withObject"
    # for i in range(1, 184):
    #     end = 1025
    #     if i == 92:
    #         end = 477
    #     for j in range(1, end):

    #         pathes.append(with_object + "/{0:04d}/{1:04d}".format(i, j))

    return pathes

def get_pose_li(dir_path):
    joint_3d = []
    joint_2d = []
    image_path = []

    for path in dir_path:
        image = path+"_color_composed.png"
        image_path.append(image)

        value = open(path+"_joint_pos.txt").readline().strip('\n').split(',')
        value = [float(val) for val in value]
        joint_3d.append(value)

        value = open(path+"_joint2D.txt").readline().strip('\n').split(',')
        value = [float(val) for val in value]
        joint_2d.append(value)

    return image_path, joint_2d, joint_3d

if __name__ == "__main__":
    # dir_path = make_dir_path('path-to-Ganerated') /home/tlh-lxy/zmh/data/GANerated_hand/GANerated/GANeratedHands_Release
    dir_path = make_dir_path('/home/tlh-lxy/zmh/data/GANerated_hand/GANerated/GANeratedHands_Release')
    image_path, joint_2d, joint_3d = get_pose_li(dir_path)

    k_means_result, k_means_centers, assignments = k_means.k_means(joint_2d, 10)

    image_path_assign = zip(image_path, assignments)
    result_dict = dict(image_path_assign)
    result_dict['centers'] = k_means_centers

    # 保存结果
    f = open('GANeratedHands_kmeans.txt','w')
    f.write(str(result_dict))
    f.close()
    
    #读取
    # f = open('GANeratedHands_kmeans.txt','r')
    # a = f.read()
    # result_dict = eval(a)
    # f.close()

    print(k_means_centers)