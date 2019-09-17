## Get Roi in model for now toy images 
import os
import cv2
import numpy as np
import matplotlib 
import math
from matplotlib import pyplot as plt

def get_rois(img_dir, show=False):
    demo_imgs_name = os.listdir(img_dir)
    demo_imgs = []
    for nm in demo_imgs_name :
        demo_imgs.append(cv2.imread(os.path.join(img_dir, nm)))

    if show :     
        for img in demo_imgs:
            plt.imshow(img)
            plt.show()

    return demo_imgs
    

def get_kp(channel, show=False):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(channel, None)
    min_resp = 99999
    max_resp = 0
    kp_filtered = [] 
    for k in kp : 
        min_resp = k.response if k.response < min_resp else min_resp
        max_resp = k.response if k.response > max_resp else max_resp

    resp_filter = 0.6 * (max_resp - min_resp)
    for k in kp:
        if k.response >= resp_filter : 
            kp_filtered.append(k)
    kp_filtered, desc = sift.compute(channel, kp_filtered)    
    if show:
        print("kp vs filter", len(kp), len(kp_filtered))

    return kp, kp_filtered, desc

def make_sift(img, show=False):
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray = np.float32(gray)
    # dst = cv2.cornerHarris(gray,2,3,0.04)
    # img[dst>0.01*dst.max()]=[255,0,255]
    # plt.imshow(img)
    # plt.show()
    # # # return(None)
    # print(img)
    # print(np.shape(img))

    im_grey = img.copy()
    kp = []; 
    kp_filtered = []; 
    desc = []
    kpc1 = get_kp(img[:,:,0])
    kpc2 = get_kp(img[:,:,1])
    kpc3 = get_kp(img[:,:,2])
    
    if kpc1[0]:
        for k in kpc1[0] : 
            kp.append(k)
        for k in kpc1[1] : 
            kp_filtered.append(k)
        for k in kpc1[2] : 
            desc.append(k)
    if kpc2[0]:
        for k in kpc2[0] : 
            kp.append(k)
        for k in kpc2[1] : 
            kp_filtered.append(k)
        for k in kpc2[2] : 
            desc.append(k)
    if kpc3[0]:
        for k in kpc3[0] : 
            kp.append(k)
        for k in kpc3[1] : 
            kp_filtered.append(k)
        for k in kpc3[2] : 
            desc.append(k)

    if show:
        print("ksp", np.shape(kp),"after filter",  np.shape(kp_filtered))
        print("desc", np.shape(desc))
        tmp_img1 = cv2.drawKeypoints(im_grey, kp, im_grey)
        f, axarr = plt.subplots(2,1)
        axarr[0].imshow(tmp_img1) 
        tmp_img2= cv2.drawKeypoints(img, kp_filtered, img)
        axarr[1].imshow(tmp_img2)
        plt.show()

    
    return kp_filtered, desc
def generate_graph(nodes, img, show=True):
    # only doing undirected graphs for now. 
    edges = []
    for node in nodes :
        for node2 in nodes:
            if is_edge(nodes, node, node2):
                if (node2,node) in edges : 
                    pass
                else:   
                    edges.append((node, node2))
            else:
                pass
    
    if show:
        for node in nodes:
            cv2.circle(img, nodes[node], 2, (0,255,255))
        for ed in edges:
             cv2.line(img, nodes[ed[0]],nodes[ed[1]],(255,0,0),1)

        plt.imshow(img)
        plt.show()

    return edges

def is_edge(nodes, a, b, dthreshold=100, nthreshold=1):
    # print(a, b)
    node_dist =  np.linalg.norm(np.array(a)-np.array(b))
    dist =  np.linalg.norm(np.array(nodes[a])-np.array(nodes[b]))
    # dist =  math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 )
    isit =  True if dist<=dthreshold and dist > 0 and node_dist<=nthreshold  else  False
    # print("is edge", dist, isit)
    return isit
    
def generate_graph_txt(img, graph, save_dir="../demo_graphs/"):
    assert(len(img) == len(graph))
    for i in img:
        file_name = os.path.join(save_dir, i)
        with open(file_name, "wb") as f:
            f.write()



def main():
    img_dir = "/home/sahmed9/Documents/reps/deep-text-recognition-benchmark/demo_image/"
    # img_dir = "/home/sahmed9/Documents/data/kenny_data_share/images_png/aa/"
    imgs = get_rois(img_dir)
    print("Imgs got  :", len(imgs))
    # kps = []
    # descs = []
    graphs = []
    # po = 5
    for po, im in enumerate(imgs) :
        kps, descs = make_sift(im)
        # kps.append(kp)
        # descs.append(desc)
        nodes = {}
        for i, kp in enumerate(kps) : 
            # nodes[i] = kp.pt
            nodes[i] = (round(kp.pt[0]), round(kp.pt[1]))
        print("NODES", len(nodes))    
        edges = generate_graph(nodes, im)
        print("Edges", len(edges))   
        graphs.append((nodes, edges, descs)) 

        if po>= 1:
            break;

    print(len(graphs[0][0]), len(graphs[0][1]), len(graphs[0][2]))

if __name__ == '__main__':
    main()