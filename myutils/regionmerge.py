import cv2
import numpy as np
import argparse
import random

def get_lab_stats(img, toLab = False):
    """
    Get the average color and min/max values for each channel in LAB color space
    """
    if toLab:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    else:
        lab = img
    l, a, b = cv2.split(lab)
    l_avg, l_min, l_max, l_std = np.mean(l), np.min(l), np.max(l), np.std(l)
    a_avg, a_min, a_max, a_std = np.mean(a), np.min(a), np.max(a), np.std(a)
    b_avg, b_min, b_max, b_std = np.mean(b), np.min(b), np.max(b), np.std(b)
    return l_avg, l_min, l_max, l_std, a_avg, a_min, a_max, a_std, b_avg, b_min, b_max, b_std

def transfer_color(img1, img2, method = 'meanstd'):
    """
    Transfer the color of img1 to img2 based on the LAB color statistics    """    

    lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
    l1, a1, b1 = cv2.split(lab1)
    l2, a2, b2 = cv2.split(lab2)

    if 'cluster' in method:
        def cluster_channel(c):
            Z = c.reshape((-1,1))
            npixels = Z.shape[0]
            npixels_max = 10000
            if npixels > npixels_max:
                #Z = cv2.resize(Z, (npixels_max,1), interpolation=cv2.INTER_NEAREST)
                Z = random.sample(list(Z), npixels_max)
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 2
            ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            return center
        
        l1_c = cluster_channel(l1)
        a1_c = cluster_channel(a1)
        b1_c = cluster_channel(b1)
        lab1_c = np.array([l1_c, a1_c, b1_c]).T
        l2_c = cluster_channel(l2)
        a2_c = cluster_channel(a2)
        b2_c = cluster_channel(b2)
        lab2_c = np.array([l2_c, a2_c, b2_c]).T

        l_avg1, l_min1, l_max1, l_std1, a_avg1, a_min1, a_max1, a_std1, b_avg1, b_min1, b_max1, b_std1 = get_lab_stats(lab1_c)
        l_avg2, l_min2, l_max2, l_std2, a_avg2, a_min2, a_max2, a_std2, b_avg2, b_min2, b_max2, b_std2 = get_lab_stats(lab2_c)
        
    elif 'boundary' in method:
        # collect image boundary pixels of img1 and img2
        l1_boundary = l1[0, :].tolist() + l1[-1, :].tolist() + l1[:, 0].tolist() + l1[:, -1].tolist()
        a1_boundary = a1[0, :].tolist() + a1[-1, :].tolist() + a1[:, 0].tolist() + a1[:, -1].tolist()
        b1_boundary = b1[0, :].tolist() + b1[-1, :].tolist() + b1[:, 0].tolist() + b1[:, -1].tolist()

        lab1_boundary = np.array([l1_boundary, a1_boundary, b1_boundary]).T
        lab1_boundary = lab1_boundary.reshape(1, lab1_boundary.shape[0], lab1_boundary.shape[1])
        l2_boundary = l2[0, :].tolist() + l2[-1, :].tolist() + l2[:, 0].tolist() + l2[:, -1].tolist()
        a2_boundary = a2[0, :].tolist() + a2[-1, :].tolist() + a2[:, 0].tolist() + a2[:, -1].tolist()
        b2_boundary = b2[0, :].tolist() + b2[-1, :].tolist() + b2[:, 0].tolist() + b2[:, -1].tolist()
        lab2_boundary = np.array([l2_boundary, a2_boundary, b2_boundary]).T
        lab2_boundary = lab2_boundary.reshape(1, lab2_boundary.shape[0], lab2_boundary.shape[1])

        l_avg1, l_min1, l_max1, l_std1, a_avg1, a_min1, a_max1, a_std1, b_avg1, b_min1, b_max1, b_std1 = get_lab_stats(lab1_boundary)
        l_avg2, l_min2, l_max2, l_std2, a_avg2, a_min2, a_max2, a_std2, b_avg2, b_min2, b_max2, b_std2 = get_lab_stats(lab2_boundary)
    else:
        l_avg1, l_min1, l_max1, l_std1, a_avg1, a_min1, a_max1, a_std1, b_avg1, b_min1, b_max1, b_std1 = get_lab_stats(lab1)
        l_avg2, l_min2, l_max2, l_std2, a_avg2, a_min2, a_max2, a_std2, b_avg2, b_min2, b_max2, b_std2 = get_lab_stats(lab2)

    # print("l_avg1 {:.2f},  l_std1 {:.2f}, a_avg1 {:.2f}, a_std1 {:.2f}, b_avg1 {:.2f}, b_std1 {:.2f}".format(l_avg1, l_std1, a_avg1, a_std1, b_avg1, b_std1))
    # print("l_avg2 {:.2f},  l_std2 {:.2f}, a_avg2 {:.2f}, a_std2 {:.2f}, b_avg2 {:.2f}, b_std2 {:.2f}".format(l_avg2, l_std2, a_avg2, a_std2, b_avg2, b_std2))

    one_thresh = 5.0
    if 'meanstd' in method:
        # Scale the color range of img1 to match that of img2, using mean-std scaling
        l1_ = (l1 - l_avg1) * (1 if l_std1 < one_thresh else float(l_std2) / float(l_std1)) + float(l_avg2)
        if 'gray' not in method:
            a1_ = (a1 - a_avg1) * (1 if a_std1 < one_thresh else float(a_std2) / float(a_std1)) + float(a_avg2)
            b1_ = (b1 - b_avg1) * (1 if b_std1 < one_thresh else float(b_std2) / float(b_std1)) + float(b_avg2)
        else:
            a1_, b1_ = a1 + float(0), b1 + float(0)
    elif 'minmax' in method:
        # Scale the color range of img1 to match that of img2, using min-max scaling
        l1_ = (l1 - l_min1) * (1 if l_std1 < one_thresh else float(l_max2 - l_min2) / float(l_max1 - l_min1)) + float(l_min2)
        if 'gray' not in method:
            a1_ = (a1 - a_min1) * (1 if a_std1 < one_thresh else float(a_max2 - a_min2) / float(a_max1 - a_min1)) + float(a_min2)
            b1_ = (b1 - b_min1) * (1 if b_std1 < one_thresh else float(b_max2 - b_min2) / float(b_max1 - b_min1)) + float(b_min2)
        else:
            a1_, b1_ = a1 + float(0), b1 + float(0)
    else:
        raise NotImplementedError
    
    # supress out of range color
    l1_ = np.minimum(np.maximum(l1_, 0),255)
    
    # Merge the LAB channels back into an image
    lab_merged = cv2.merge((l1_, a1_, b1_))
    img_transferred = cv2.cvtColor(lab_merged.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return img_transferred

def get_clean_boundary_pixels(img_):
    w, h = img_.shape[1], img_.shape[0]

    # copy nearest pixel from boundary and calculate weight according to distance to boundary
    img_ne = img_.copy()

    # cluster pixels of img1
    img_lab = cv2.cvtColor(img_.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l1, a1, b1 = cv2.split(img_lab)
    l1_c = l1.reshape((-1,1))
    a1_c = a1.reshape((-1,1))
    b1_c = b1.reshape((-1,1))
    Z = np.concatenate((l1_c, a1_c, b1_c), axis=1)
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    label = label.reshape((img_.shape[0], img_.shape[1]))
    nlabel1 = np.sum(label[0,:]) + np.sum(label[-1,:]) + np.sum(label[:,0]) + np.sum(label[:,-1])
    if nlabel1 > (w + h + 2):
        bgLabel = 1
    else:
        bgLabel = 0
    
    # replace non-bg pixels on boundary with nearest bg pixel
    label_ = label.copy()
    for i in range(1, w):
        for j in range(1, h):
            if label_[j,i] != bgLabel:
                if label_[j,i-1] == bgLabel:
                    img_ne[j,i] = img_ne[j,i-1]
                elif label_[j-1,i] == bgLabel:
                    img_ne[j,i] = img_ne[j-1,i]        
    for i in range(w-2, -1, -1):
        for j in range(h-2, -1, -1):
            if label_[j,i] != bgLabel:
                if label_[j,i+1] == bgLabel:
                    img_ne[j,i] = img_ne[j,i+1]
                elif label_[j+1,i] == bgLabel:
                    img_ne[j,i] = img_ne[j+1,i]

    return img_ne

def merge(img1, img2, bbox, method = ''):
    # crop part of img1 and fill it into the same part of img2
    img2_merged = img2.copy()
    x, y, w, h = bbox

    if 'smooth' in method:
        img2_ = img2[y:y+h, x:x+w].astype(np.float32)
        img1_ = img1.astype(np.float32)

        # copy nearest pixel from boundary and calculate weight according to distance to boundary
        if 'cleanedge' in method:
            img1_ne = get_clean_boundary_pixels(img1_)
            img2_ne = get_clean_boundary_pixels(img2_)
        else:
            img1_ne = img1_.copy()
            img2_ne = img2_.copy()

        weight = np.zeros((h,w,3), dtype=np.float32)
        for i in range(w):
            for j in range(h):
                if i != 0 and j != 0 and i != w-1 and j != h-1:
                    min_dist = min(i, j, w-i-1, h-j-1)
                    weight[j,i] = min_dist
                    if i == min_dist:
                        img1_ne[j,i] = img1_ne[j,i-1]
                        img2_ne[j,i] = img2_ne[j,i-1]
                    elif j == min_dist:
                        img1_ne[j,i] = img1_ne[j-1,i]
                        img2_ne[j,i] = img2_ne[j-1,i]
                    
        for i in range(w-1,0,-1):
            for j in range(h-1,0,-1):
                if i != 0 and j != 0 and i != w-1 and j != h-1:
                    min_dist = min(i, j, w-i-1, h-j-1)
                    if w-i-1 == min_dist:
                        img1_ne[j,i] = img1_ne[j,i+1]
                        img2_ne[j,i] = img2_ne[j,i+1]
                    elif h-j-1 == min_dist:
                        img1_ne[j,i] = img1_ne[j+1,i]
                        img2_ne[j,i] = img2_ne[j+1,i]
                
        dist_decay, color_decay = 50, 50
        weight_dist = np.exp(-weight/dist_decay)
        weight_color1 = np.exp(-np.sqrt(np.sum(np.multiply((img1_-img1_ne), (img1_-img1_ne)), 2)) / color_decay)
        weight_color2 = np.exp(-np.sqrt(np.sum(np.multiply((img2_-img2_ne), (img2_-img2_ne)), 2)) / color_decay)
        weight_color = np.multiply(weight_color1, weight_color2)
        weight_color = np.repeat(weight_color[:,:,np.newaxis], 3, axis=2)
        weight = np.multiply(weight_dist, weight_color)
        # cv2.imwrite("weight.png", (weight * 255).astype(np.uint8))

        img2_merged[y:y+h, x:x+w] = np.multiply(img2[y:y+h, x:x+w], weight) + np.multiply(img1, 1-weight)
        img2_merged = img2_merged.astype(np.uint8)
    else:
        img2_merged[y:y+h, x:x+w] = img1

    return img2_merged


if __name__ == '__main__':
    # parse image
    parser = argparse.ArgumentParser()
    #parser.add_argument('--img1', type=str, default='2/gan.png')
    #parser.add_argument('--img2', type=str, default='2/pasd.png')
    #parser.add_argument('--ref', type=str, default='2/org.jpeg')
    #parser.add_argument('--img1_adjust', type=str, default='2/gan_adjust.png')
    #parser.add_argument('--bbox', type=str, default='0.1,0.5,0.8,0.4')
    parser.add_argument('--img1', type=str, required=True)
    parser.add_argument('--img2', type=str, required=True)
    parser.add_argument('--ref', type=str)
    parser.add_argument('--img1_adjust', type=str, required=True)
    parser.add_argument('--bbox', type=str, required=True)
    args = parser.parse_args()

    # read image
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    assert(img1.shape == img2.shape)

    # calculate bbox
    bbox = [float(i) for i in args.bbox.split(',')]
    w, h = img2.shape[1], img2.shape[0]
    if np.max(bbox) <= 1:
        x, y, w, h = int(w*bbox[0]), int(h*bbox[1]), int(w*bbox[2]), int(h*bbox[3])
    else:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    bbox = [x,y,w,h]

    # transfer color to reference image
    if args.ref is not None:
        ref = cv2.imread(args.ref)
        ref = cv2.resize(ref, (w,h))
        img1 = transfer_color(img1, ref, "meanstd")
        img2 = transfer_color(img2, ref, "meanstd")

    # roi region
    img1_crop = img1[y:y+h, x:x+w]
    img2_crop = img2[y:y+h, x:x+w]

    # transfer color
    img_transferred = transfer_color(img1_crop, img2_crop, "meanstd-cluster")
    img_merged_boundary = merge(img_transferred, img2, bbox)
    cv2.imwrite(args.img1_adjust+".cluster.png", img_merged_boundary)

    img_transferred = transfer_color(img1_crop, img2_crop, "meanstd-boundary")
    img_merged_boundary = merge(img_transferred, img2, bbox)
    cv2.imwrite(args.img1_adjust+".boundary.png", img_merged_boundary)

    img_transferred = transfer_color(img1_crop, img2_crop, "meanstd")
    img_merged_meanstd = merge(img_transferred, img2, bbox)
    cv2.imwrite(args.img1_adjust+".meanstd.png", img_merged_meanstd)
    img_merged_smooth = merge(img_transferred, img2, bbox, method = 'smooth')
    cv2.imwrite(args.img1_adjust+".smooth.png", img_merged_smooth)
    img_merged_smooth_cleanedge = merge(img_transferred, img2, bbox, method = 'smooth-cleanedge')
    cv2.imwrite(args.img1_adjust+".smooth-cleanedge.png", img_merged_smooth_cleanedge)
    
    img_transferred = transfer_color(img1_crop, img2_crop, "meanstd-gray")
    img_merged_meanstd_gray = merge(img_transferred, img2, bbox)
    cv2.imwrite(args.img1_adjust+".meanstd-gray.png", img_merged_meanstd_gray)

    img_merged_org = merge(img1[y:y+h, x:x+w], img2, bbox)
    cv2.imwrite(args.img1_adjust+".org.png", img_merged_org)
